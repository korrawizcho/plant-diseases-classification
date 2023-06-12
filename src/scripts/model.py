import torch
from torch import nn
import hydra
from hydra.core.config_store import ConfigStore
import os 
from constant import ROOT_DIR, CONFIG_FOLDER, CONFIG_NAME
import constructor


from hydra import compose, initialize
from omegaconf import OmegaConf

hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(config_path='../../config', job_name="config")
cfg = compose(config_name=CONFIG_NAME)


def init_model(cfg: constructor.Config = cfg, pretrain: bool = True):
    model = torch.hub.load(str(cfg.model_config.hub_url),
                           str(cfg.model_config.model_name),
                           pretrained=pretrain)
    for param in model.parameters():  # freeze model
        param.requires_grad = False

    n_inputs = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, cfg.dataloader_config.num_classes)
    )
    return model


def get_device(cfg: constructor.Config = cfg):
    return torch.device(cfg.model_config.device)


if __name__ == '__main__':
    print(init_model())

    get_device()