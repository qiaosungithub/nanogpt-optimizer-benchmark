"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
import json
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

from foof import FOOF, FoofConfig, HookedLinear, collect_foof_named_params
from foof import Muon, MuonConfig
from logging_util import ExperimentLogger


def load_foof_config() -> FoofConfig:
    raw = os.environ.get("FOOF_CONFIG")
    if raw is None:
        return FoofConfig()
    data = json.loads(raw)
    return FoofConfig(**data)


def load_muon_config() -> MuonConfig:
    raw = os.environ.get("MUON_CONFIG")
    if raw is None:
        return MuonConfig()
    data = json.loads(raw)
    return MuonConfig(**data)


def load_matrix_optimizer_name() -> str:
    return os.environ.get("MATRIX_OPT", "foof").strip().lower()


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def grad_l2_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad.detach().float()
        total += float(grad.square().sum().item())
    return math.sqrt(total)


def collect_lrs(optimizer1: AdamW, optimizer2) -> dict[str, float]:
    return {
        "lr/adam_embed": float(optimizer1.param_groups[0]["lr"]),
        "lr/adam_proj": float(optimizer1.param_groups[1]["lr"]),
        "lr/adam_other": float(optimizer1.param_groups[2]["lr"]),
        "lr/matrix": float(optimizer2.param_groups[0]["lr"]),
    }


def _hooked_linear_stats(module: HookedLinear) -> tuple[float, float, float] | None:
    if module._input_mean is None or module._input_cov_full is None:
        return None
    mean = module._input_mean
    cov = module._input_cov_full
    dim = mean.numel()
    mean_abs = mean.abs().mean().item()
    second_moment = (torch.trace(cov) / dim).item()
    mean_sq = mean.square().mean().item()
    var = max(0.0, second_moment - mean_sq)
    return mean_abs, math.sqrt(max(0.0, second_moment)), math.sqrt(var)


def collect_activation_metrics(model) -> dict[str, float]:
    metrics = {}
    points = {
        "act/attn_q": model.blocks[0].attn.q,
        "act/attn_proj": model.blocks[0].attn.proj,
        "act/mlp_fc": model.blocks[0].mlp.fc,
        "act/mlp_proj": model.blocks[0].mlp.proj,
    }
    for prefix, module in points.items():
        stats = _hooked_linear_stats(module)
        if stats is None:
            continue
        mean_abs, rms, std = stats
        metrics[f"{prefix}/mean_abs"] = mean_abs
        metrics[f"{prefix}/rms"] = rms
        metrics[f"{prefix}/std"] = std
    return metrics


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = HookedLinear(dim, hdim)
        self.k = HookedLinear(dim, hdim)
        self.v = HookedLinear(dim, hdim)
        self.proj = HookedLinear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = HookedLinear(dim, hdim)
        self.proj = HookedLinear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

seed = env_int("SEED", 1337)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
log_per_step = env_int("LOG_PER_STEP", 100)
train_print_every = env_int("TRAIN_PRINT_EVERY", 10)
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


num_trials = 1
if len(sys.argv) > 1 and sys.argv[-1].strip() != "":
    num_trials = int(sys.argv[-1])
matrix_opt = load_matrix_optimizer_name()
train_steps = 3375
run_config = {
    "seed": seed,
    "matrix_opt": matrix_opt,
    "world_size": dist.get_world_size(),
    "batch_size": batch_size,
    "mbs": mbs,
    "train_steps": train_steps,
    "val_every": 125,
    "log_per_step": log_per_step,
    "train_print_every": train_print_every,
    "num_trials": num_trials,
    "foof_config": os.environ.get("FOOF_CONFIG"),
    "muon_config": os.environ.get("MUON_CONFIG"),
}
logger = ExperimentLogger(dist.get_rank(), log_per_step, run_config)

for trial_idx in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s)
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    if matrix_opt == "foof":
        foof_config = load_foof_config()
        optimizer2 = FOOF(collect_foof_named_params(model), config=foof_config)
    elif matrix_opt == "muon":
        muon_config = load_muon_config()
        optimizer2 = Muon(
            [p for p in model.blocks.parameters() if p.ndim >= 2],
            config=muon_config,
        )
    else:
        raise ValueError(f"Unsupported MATRIX_OPT={matrix_opt}. Expected 'foof' or 'muon'.")
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # start the clock
    training_time = 0
    last_val_step = 0
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(train_steps + 1):
        global_step = trial_idx * (train_steps + 1) + step

        # --------------- VALIDATION SECTION -----------------
        if step == train_steps or step % 125 == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms", console=True)
            logger.log(
                global_step,
                {
                    "trial": trial_idx,
                    "val/loss": val_loss,
                    "time/train_seconds": training_time,
                    "time/val_interval_ms": 1000 * step_avg,
                    "progress/step": step,
                },
                force=(step == train_steps),
            )
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        should_log_step = logger.should_log(global_step)
        train_loss = None
        train_loss_local = torch.zeros((), device=device, dtype=torch.float32)
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
            if should_log_step:
                train_loss_local += loss.detach().float()
            loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        if should_log_step:
            dist.all_reduce(train_loss_local, op=dist.ReduceOp.SUM)
            train_loss = (train_loss_local / batch_size).item()
        # set optimization hyperparameters and take a step
        set_hparams(step)
        if should_log_step:
            log_metrics = {
                "trial": trial_idx,
                "train/loss": train_loss,
                "grad/global_l2": grad_l2_norm(model.parameters()),
                "grad/matrix_l2": grad_l2_norm(p for p in model.blocks.parameters() if p.ndim >= 2),
                "progress/step": step,
                "progress/tokens_seen": (trial_idx * train_steps + step + 1) * batch_size,
            }
            log_metrics.update(collect_lrs(optimizer1, optimizer2))
            log_metrics.update(collect_activation_metrics(model))
            logger.log(global_step, log_metrics)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        if (step + 1) % train_print_every == 0:
            print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
                   + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

logger.finish()
dist.destroy_process_group()
