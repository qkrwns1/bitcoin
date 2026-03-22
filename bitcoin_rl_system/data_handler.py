"""Data loading and feature preparation scaffold.

Planned responsibilities:
- load processed 1-minute Bitcoin data
- define sequence/context feature groups
- split data by time into train/validation/test
- provide RL-ready market frames
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    data_root: Path = Path(__file__).resolve().parent.parent / "data analysis" / "data"
    market: str = "KRW-BTC"
    step_minutes: int = 1
    sequence_length: int = 360
    context_feature_names: list[str] = field(default_factory=list)
    sequence_feature_names: list[str] = field(default_factory=list)


class BitcoinDataHandler:
    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def load_processed_frames(self) -> None:
        raise NotImplementedError("Processed frame loading is not implemented yet.")

    def build_feature_layout(self) -> dict[str, list[str]]:
        return {
            "sequence": self.config.sequence_feature_names,
            "context": self.config.context_feature_names,
        }

    def split_by_time(self) -> None:
        raise NotImplementedError("Temporal split logic is not implemented yet.")
