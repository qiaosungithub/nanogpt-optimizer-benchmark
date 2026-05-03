import os
from typing import Any


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _to_scalar(value: Any) -> float | int | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (int, float)):
        return value
    return None


class ExperimentLogger:
    def __init__(self, rank: int, log_every: int, run_config: dict[str, Any]):
        self.rank = rank
        self.log_every = max(1, int(log_every))
        self.enabled = rank == 0 and _env_flag("WANDB_ENABLE", "1")
        self._wandb = None
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError:
            print("wandb is not installed; disabling WANDB logging")
            self.enabled = False
            return

        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)

        project = os.environ.get("WANDB_PROJECT", "nanogpt-optimizer-benchmark")
        entity = os.environ.get("WANDB_ENTITY")
        tags = [t.strip() for t in os.environ.get("WANDB_TAGS", "").split(",") if t.strip()]
        run_name = os.environ.get("WANDB_RUN_NAME")
        wandb.init(
            project=project,
            entity=entity or None,
            name=run_name,
            tags=tags,
            config=run_config,
        )
        self._wandb = wandb

    def should_log(self, step: int) -> bool:
        return step % self.log_every == 0

    def log(self, step: int, metrics: dict[str, Any], force: bool = False):
        if not self.enabled or self._wandb is None:
            return
        if not force and not self.should_log(step):
            return
        payload = {}
        for key, value in metrics.items():
            scalar = _to_scalar(value)
            if scalar is not None:
                payload[key] = scalar
        if payload:
            self._wandb.log(payload, step=step)

    def finish(self):
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
