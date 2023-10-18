import math
from logging import getLogger


logger = getLogger(__name__)


class CosineScheduler:
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        steps: int,
        zerolr_warmup_steps: int,
        linear_warmup_steps: int,
        cooldown_steps: int,
    ) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.steps = steps
        self.zerolr_warmup_steps = zerolr_warmup_steps
        self.linear_warmup_steps = linear_warmup_steps
        self.cooldown_steps = cooldown_steps

    def __call__(self, step: int) -> float:
        if step < self.zerolr_warmup_steps:
            return 0.0
        step -= self.zerolr_warmup_steps

        if step < self.linear_warmup_steps:
            return self.max_lr * step / self.linear_warmup_steps
        step -= self.linear_warmup_steps

        cosine_steps = (
            self.steps - self.zerolr_warmup_steps - self.linear_warmup_steps - self.cooldown_steps
        )
        if step < cosine_steps:
            return (
                self.min_lr
                + (self.max_lr - self.min_lr) * (1 + math.cos(step / cosine_steps * math.pi)) / 2
            )
        step -= cosine_steps

        if step < self.cooldown_steps:
            return self.min_lr * (self.cooldown_steps - step) / self.cooldown_steps
        else:
            logger.warning(f"Step is out of range: {step}")
            return 0.0
