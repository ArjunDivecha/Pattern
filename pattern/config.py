"""
=============================================================================
SCRIPT NAME: config.py
=============================================================================
DESCRIPTION:
Pydantic configuration models matching the YAML schema in PRD §10.
Load with Config.from_yaml("configs/debug.yaml").
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, computed_field


class DataConfig(BaseModel):
    csv_path: Path = Path("./data/r1000_ohlcv_database.csv")
    date_format: str = "%Y-%m-%d"
    min_history_days: int = 252
    exclude_lifecycle_boundaries: bool = False


class ImageConfig(BaseModel):
    window: int = 20
    height: int = 64
    width: int = 60             # must equal 3 * window
    ohlc_height_ratio: float = 0.797   # 51/64 ≈ 0.797
    include_ma: bool = True
    include_volume: bool = True
    cache_dir: Path = Path("./cache/images_I20")

    @computed_field  # type: ignore[misc]
    @property
    def ohlc_rows(self) -> int:
        return int(round(self.height * self.ohlc_height_ratio))

    @computed_field  # type: ignore[misc]
    @property
    def vol_rows(self) -> int:
        return self.height - self.ohlc_rows - 1  # 1 gap row


class LabelConfig(BaseModel):
    horizon: int = 20
    balance_train: bool = True


class ModelConfig(BaseModel):
    blocks: int = 3
    channels: List[int] = [64, 128, 256]
    conv_kernel: List[int] = [5, 3]

    # First-block convolution params. Per paper §II.B: the asymmetric vertical
    # stride and dilation exist to down-sample the sparse first layer, and are
    # applied ONLY on the first block. Deeper blocks keep kernel (5,3) but use
    # stride=1, dilation=1, same-padding.
    conv_stride:   List[int] = [3, 1]    # first block only
    conv_padding:  List[int] = [12, 1]   # first block only
    conv_dilation: List[int] = [2, 1]    # first block only

    # Deeper-block convolution params (blocks 2+). Same-padding so spatial dims
    # only change via the 2×1 max-pool between blocks.
    conv_stride_inner:   List[int] = [1, 1]
    conv_padding_inner:  List[int] = [2, 1]
    conv_dilation_inner: List[int] = [1, 1]

    pool_kernel: List[int] = [2, 1]
    leaky_slope: float = 0.01
    fc_dropout: float = 0.5


class TrainConfig(BaseModel):
    ensemble_size: int = 5
    batch_size: int = 128
    lr: float = 1e-5
    optimizer: str = "adam"
    max_epochs: int = 100
    early_stop_patience: int = 2
    seeds: List[int] = [0, 1, 2, 3, 4]
    device: str = "auto"
    num_workers: int = 4
    use_wandb: bool = False
    wandb_project: str = "pattern-cnn"


class SplitConfig(BaseModel):
    mode: Literal["debug", "expanding", "rolling"] = "debug"
    start_year: Optional[int] = None
    train_years: int = 3
    val_fraction: float = 0.30
    test_years: int = 2
    retrain_every_years: int = 1
    # Rolling mode only: train window grows from `train_years` up to
    # `max_train_years` (inclusive), then keeps trailing `max_train_years`.
    # None → legacy fixed-width rolling at `train_years`.
    max_train_years: Optional[int] = None


class BacktestConfig(BaseModel):
    n_deciles: int = 10
    weighting: List[str] = ["equal", "value"]
    holding_period_days: int = 20
    newey_west_lags: int = 4


class Config(BaseModel):
    data: DataConfig = DataConfig()
    image: ImageConfig = ImageConfig()
    label: LabelConfig = LabelConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    split: SplitConfig = SplitConfig()
    backtest: BacktestConfig = BacktestConfig()
    output_dir: Path = Path("./runs")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw or {})
