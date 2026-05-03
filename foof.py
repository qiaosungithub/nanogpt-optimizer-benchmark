from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist


class HookedLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
        self._input_mean = None
        self._input_cov_full = None

    def forward(self, x):
        x2d = x.reshape(-1, x.shape[-1]).detach().to(torch.float32)
        n = max(1, x2d.shape[0])
        self._input_mean = x2d.mean(dim=0)
        self._input_cov_full = (x2d.mT @ x2d) / n
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


@dataclass
class FoofConfig:
    lr: float = 0.025
    weight_decay: float = 0.025
    beta: float = 0.95
    fw_steps: int = 4
    alpha_mult: float = 1.0
    nesterov: bool = True
    eps: float = 1e-12


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def _surrogate_inv(M: Tensor, mean: Tensor, sigma_sq: Tensor, eps: float = 1e-12) -> Tensor:
    sigma_sq_safe = sigma_sq + eps
    mu_norm_sq = torch.dot(mean, mean)
    denom = sigma_sq_safe + mu_norm_sq
    mu_t_m = mean @ M
    correction = torch.outer(mean, mu_t_m) / denom
    return (M - correction) / sigma_sq_safe


@torch.compile
def foof_update(
    grad: Tensor,
    momentum: Tensor,
    cov_ema: Tensor,
    mean_ema: Tensor,
    input_cov: Tensor,
    input_mean: Tensor,
    beta: float,
    nesterov: bool,
    fw_steps: int,
    alpha_mult: float,
    eps: float,
):
    momentum.lerp_(grad, 1 - beta)
    cov_ema.lerp_(input_cov, 1 - beta)
    mean_ema.lerp_(input_mean, 1 - beta)

    m_t = grad.lerp_(momentum, beta) if nesterov else momentum
    d, m_dim = m_t.shape
    mean_norm_sq = torch.dot(mean_ema, mean_ema)
    tr_cov = torch.trace(cov_ema)
    sigma_sq = torch.clamp((tr_cov - mean_norm_sq) / d, min=eps)

    p_t = _surrogate_inv(m_t, mean_ema, sigma_sq, eps=eps)
    p_norm = torch.norm(p_t)
    alpha_t = alpha_mult / (p_norm / (min(d, m_dim) ** 0.5) + eps)
    t_t = alpha_t * m_t

    u = torch.zeros_like(t_t)
    for _ in range(fw_steps):
        r = t_t - cov_ema @ u
        s = zeropower_via_newtonschulz5(r)
        d_t = s - u
        num = torch.sum(r * d_t)
        cd = cov_ema @ d_t
        den = torch.sum(d_t * cd) + eps
        gamma = torch.clamp(num / den, 0.0, 1.0)
        u = u + gamma * d_t

    u *= max(1, grad.size(-1) / grad.size(-2)) ** 0.5
    return u


def collect_foof_named_params(model):
    out = []
    for module_name, module in model.named_modules():
        if isinstance(module, HookedLinear) and module_name.startswith("blocks"):
            out.append((module_name, module.weight))
    return out


class FOOF(torch.optim.Optimizer):
    def __init__(self, named_params, config: FoofConfig):
        named_params = list(named_params)
        assert len(named_params) >= 1
        params = [p for _, p in named_params]
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        self.param_to_module = {p: module for module, p in named_params}
        defaults = dict(
            lr=config.lr,
            weight_decay=config.weight_decay,
            beta=config.beta,
            fw_steps=config.fw_steps,
            alpha_mult=config.alpha_mult,
            nesterov=config.nesterov,
            eps=config.eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                        state["cov_ema"] = torch.eye(p.size(0), device=p.device, dtype=torch.float32)
                        state["mean_ema"] = torch.zeros(p.size(0), device=p.device, dtype=torch.float32)

                    owner = self.param_to_module[p]
                    if owner._input_cov_full is None:
                        update = muon_update(
                            p.grad,
                            state["momentum"],
                            mu=group["beta"],
                            nesterov=group["nesterov"],
                        )
                    else:
                        input_cov = owner._input_cov_full.to(device=p.device, dtype=torch.float32)
                        input_mean = owner._input_mean.to(device=p.device, dtype=torch.float32)
                        update = foof_update(
                            p.grad,
                            state["momentum"],
                            state["cov_ema"],
                            state["mean_ema"],
                            input_cov,
                            input_mean,
                            beta=group["beta"],
                            nesterov=group["nesterov"],
                            fw_steps=group["fw_steps"],
                            alpha_mult=group["alpha_mult"],
                            eps=group["eps"],
                        )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
