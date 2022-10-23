import dataclasses as dt
import pathlib as pl
import typing as t
from enum import Enum

from pgt import Config as pGTConfig

# path to the root of the source code
root = pl.Path(__file__).parent.parent.absolute()


@dt.dataclass()
class NetworkConfig:
    num_stages: int = 4
    num_layers: int = 10
    num_f_maps: int = 64
    features_dim: int = 2048
    bz: int = 8
    lr: float = 0.0005
    wd: float = 0.0
    num_epochs: int = 50


@dt.dataclass()
class PathConfig:
    data_root: str = str(root / "data")
    models_root: str = str(root / "models")
    results_root: str = str(root / "results")
    logs_root: str = str(root / "logs")
    annotation_root: str = str(root / "timestamps_annotations")


class pGTType(Enum):
    baseline = 0
    hard = 1
    oracle = 2


@dt.dataclass()
class LossConfig:
    smoothing_factor: float = 0.15
    smoothing_clamp_max: int = 16
    confidence_factor: float = 0.075


@dt.dataclass()
class Config:
    seed: int = 1
    device: str = "cuda"
    action: str = "train"
    dataset: str = "50salads"  # breakfast, 50salads
    split: str = "1"
    timestamp_percentage: int = 95
    notes: str = "EXPERIMENT"
    network: NetworkConfig = NetworkConfig()
    path: PathConfig = PathConfig()
    pgt_type: pGTType = pGTType.hard
    pgt_config: pGTConfig = pGTConfig()
    loss: LossConfig = LossConfig()
    pgt_training_start_at: int = 30
    no_save_no_writer: bool = True
    annotation_file: t.Optional[str] = None
