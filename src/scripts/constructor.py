from pydantic.dataclasses import dataclass
from typing import List
from torch.utils.data import DataLoader


@dataclass
class DataLoaderConfig:
    data_path: str
    train_batch: int
    test_batch: int
    num_workers: int
    num_classes: int

@dataclass
class ModelConfig:
    num_epochs: int
    hub_url: str
    model_name: str
    learning_rate: float
    weight_decay: float
    step_per_epoch: int
    checkpoint_path: str
    checkpoint_name: str
    csv_logger_path: str
    device: str


@dataclass
class Config:
    dataloader_config: DataLoaderConfig
    model_config: ModelConfig


