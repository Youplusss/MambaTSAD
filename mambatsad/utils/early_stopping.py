# utils/early_stopping.py
# 中文注释版

from dataclasses import dataclass

@dataclass
class EarlyStopping:
    patience: int = 10        # 连续多少个 epoch 指标不提升就停止
    min_delta: float = 0.0    # 提升至少要超过这个值才算“变好”
    mode: str = "max"         # "max" 监控 F1/AUC, "min" 监控 loss

    def __post_init__(self):
        if self.mode not in ("max", "min"):
            raise ValueError("mode 必须是 'max' 或 'min'")
        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, metric_value: float) -> bool:
        """
        更新早停器状态
        返回 True 表示应该停止训练
        """
        if self.best is None:
            self.best = metric_value
            self.num_bad_epochs = 0
            return False

        if self.mode == "max":
            improvement = metric_value - self.best
        else:
            improvement = self.best - metric_value

        if improvement > self.min_delta:
            self.best = metric_value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.should_stop = True

        return self.should_stop
