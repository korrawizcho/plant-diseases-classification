import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import torch
from torchvision import datasets
from torchvision import transforms as T # for simplifying the transforms
from torch.utils.data import DataLoader
import timm
import hydra
from hydra.core.config_store import ConfigStore
import os 
from constructor import Config
from constant import ROOT_DIR, CONFIG_FOLDER, CONFIG_NAME, cfg


from hydra import compose, initialize
from omegaconf import OmegaConf



def get_classes():
    all_data = datasets.ImageFolder(
        os.path.join(ROOT_DIR, cfg.dataloader_config.data_path, "train/")
    )
    return all_data.classes


def get_data_loaders(train=False):
    data_dir = os.path.join(ROOT_DIR, cfg.dataloader_config.data_path)
    batch_size = cfg.dataloader_config.train_batch
    num_workers = cfg.dataloader_config.num_workers

    if train:
        #train
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
            T.RandomErasing(p=0.1, value='random')
        ])
        train_data = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform = transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return train_loader, len(train_data)

    else:
        # val/test
        transform = T.Compose([ # We dont need augmentation for test transforms
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
        ])

        val_data = datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)
        test_data = datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return val_loader, test_loader, len(val_data), len(test_data)

if __name__ == '__main__':
    get_classes()
    print('dataloader type', get_data_loaders(train=True))
