import os
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import mlflow.pytorch
from mlflow import MlflowClient
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from torchmetrics.classification import MulticlassAccuracy
import torch
from constant import ROOT_DIR, CONFIG_FOLDER, CONFIG_NAME
from constructor import Config
from model import init_model
from dataloader import get_data_loaders
from timm.loss import LabelSmoothingCrossEntropy 
from torch import optim
import hydra

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./mlruns")
trainer = Trainer(logger=mlf_logger)

from hydra import compose, initialize
from omegaconf import OmegaConf


cfg = compose(config_name=CONFIG_NAME)
# cfg = OmegaConf.to_yaml(cfg)
# print(cfg.dataloader_config.train_batch)



class PlantDiseasesModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.model = init_model()
        self.criterion = LabelSmoothingCrossEntropy()
        

    def forward(self, x):
        # self.model.train()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = MulticlassAccuracy(
            num_classes=3,
            average="macro"
            )(preds, y)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    

    def configure_optimizers(self):
        weight_decay = float(self.cfg.model_config.weight_decay)
        optimizer = optim.AdamW(
            self.model.head.parameters(),
            lr=self.cfg.model_config.learning_rate,
            weight_decay=weight_decay,
        )

        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=45000 // 1,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict} 


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    # save metrics to results folder
    import json
    with open(os.path.join(ROOT_DIR, "results", "metrics.json"), "w") as f:
        json.dump(r.data.metrics, f)
    print("tags: {}".format(tags))



def init_trainer(cfg):
    model = PlantDiseasesModel()
    checkpoint_dir = os.path.join(ROOT_DIR, "models", cfg.model_config.checkpoint_path)
    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=cfg.model_config.checkpoint_name,
        save_top_k=1,
        verbose=True,
        auto_insert_metric_name=True)
    trainer = pl.Trainer(
        max_epochs=cfg.model_config.num_epochs,
        # accelerator="auto",
        logger=CSVLogger(save_dir=ROOT_DIR, name="lightning_logs", version=1),
        callbacks=[checkpoint_callback,
                   LearningRateMonitor(logging_interval="step"),
                   TQDMProgressBar(refresh_rate=10)],
    )
    return trainer, model


def start_training():
    trainer, model = init_trainer(cfg)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    mlflow.pytorch.autolog()
    # Train the model
    with mlflow.start_run() as run:
        trainer.fit(model,
                    train_dataloaders=get_data_loaders(train=True)[0],
                    val_dataloaders=get_data_loaders(train=False)[0])

        # fetch the auto logged parameters and metrics
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    start_training()