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
    fw_ns_variant: str = "poly5"
    fw_ns_steps: int = 12


@dataclass
class MuonConfig:
    lr: float = 0.025
    weight_decay: float = 0.025
    mu: float = 0.95
    nesterov: bool = True


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


def zeropower_via_newtonschulz_basic(G: Tensor, steps: int = 12) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = 1.5 * X - 0.5 * (A @ X)
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def ns_orthogonalize(G: Tensor, variant_id: int, steps: int) -> Tensor:
    if variant_id == 0:
        return zeropower_via_newtonschulz5(G)
    if variant_id == 1:
        return zeropower_via_newtonschulz_basic(G, steps=steps)
    raise ValueError(f"Unsupported NS variant id: {variant_id}. Expected 0(poly5) or 1(basic).")


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
    fw_ns_variant_id: int,
    fw_ns_steps: int,
):
    momentum.lerp_(grad, 1 - beta)
    cov_ema.lerp_(input_cov, 1 - beta)
    mean_ema.lerp_(input_mean, 1 - beta)

    m_raw = grad.lerp_(momentum, beta) if nesterov else momentum
    M_t = m_raw.mT
    d, m_dim = M_t.shape
    mean_norm_sq = torch.dot(mean_ema, mean_ema)
    tr_cov = torch.trace(cov_ema)
    sigma_sq = torch.clamp((tr_cov - mean_norm_sq) / d, min=eps)

    p_t = _surrogate_inv(M_t, mean_ema, sigma_sq, eps=eps)
    p_norm = torch.norm(p_t)
    alpha_t = alpha_mult / (p_norm / (min(d, m_dim) ** 0.5) + eps)
    t_t = alpha_t * M_t

    u = torch.zeros_like(t_t)
    for _ in range(fw_steps):
        r = t_t - cov_ema @ u
        s = ns_orthogonalize(r, fw_ns_variant_id, fw_ns_steps)
        d_t = s - u
        num = torch.sum(r * d_t)
        cd = cov_ema @ d_t
        den = torch.sum(d_t * cd) + eps
        gamma = torch.clamp(num / den, 0.0, 1.0)
        u = u + gamma * d_t

    update = u.mT
    update *= max(1, m_dim / d) ** 0.5
    return update


def collect_foof_named_params(model):
    out = []
    for _, module in model.named_modules():
        if isinstance(module, HookedLinear):
            out.append((module, module.weight))
    return out


class FOOF(torch.optim.Optimizer):
    def __init__(self, named_params, config: FoofConfig):
        named_params = list(named_params)
        assert len(named_params) >= 1
        if config.fw_ns_variant not in ("poly5", "basic"):
            raise ValueError("FoofConfig.fw_ns_variant must be 'poly5' or 'basic'")
        fw_ns_variant_id = 0 if config.fw_ns_variant == "poly5" else 1
        params = [p for _, p in named_params]
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        self.param_to_module = {p: module for module, p in named_params}
        for module, _ in named_params:
            if not isinstance(module, HookedLinear):
                raise TypeError("FOOF expects (HookedLinear, parameter) pairs")
        defaults = dict(
            lr=config.lr,
            weight_decay=config.weight_decay,
            beta=config.beta,
            fw_steps=config.fw_steps,
            alpha_mult=config.alpha_mult,
            nesterov=config.nesterov,
            eps=config.eps,
            fw_ns_variant_id=fw_ns_variant_id,
            fw_ns_steps=config.fw_ns_steps,
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
                        in_dim = p.size(1)
                        state["cov_ema"] = torch.eye(in_dim, device=p.device, dtype=torch.float32)
                        state["mean_ema"] = torch.zeros(in_dim, device=p.device, dtype=torch.float32)

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
                            fw_ns_variant_id=group["fw_ns_variant_id"],
                            fw_ns_steps=group["fw_ns_steps"],
                        )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


class Muon(torch.optim.Optimizer):
    def __init__(self, params, config: MuonConfig):
        params = list(params)
        assert len(params) >= 1
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(
            lr=config.lr,
            weight_decay=config.weight_decay,
            mu=config.mu,
            nesterov=config.nesterov,
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
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum"],
                        mu=group["mu"],
                        nesterov=group["nesterov"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
