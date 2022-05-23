import argparse
import collections
from typing import Optional, Union, Dict

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only


class MetricLogger(LightningLoggerBase):
    def __init__(self) -> None:
        super().__init__()
        self.history = collections.defaultdict(list)

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        for metric_name, metric_value in metrics.items():
            if metric_name != 'epoch':
                self.history[metric_name].append(metric_value)
            else:
                if (
                    not len(self.history['epoch']) or
                    not self.history['epoch'][-1] == metric_value
                ):
                    self.history['epoch'].append(metric_value)

    def log_hyperparams(
        self, params: argparse.Namespace, *args, **kwargs
    ) -> None:
        pass

    @property
    def name(self) -> str:
        return "MetricLogger"

    @property
    def version(self) -> Union[int, str]:
        return "0.1"
