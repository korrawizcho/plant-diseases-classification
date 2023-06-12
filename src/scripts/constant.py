from pathlib import Path
from hydra.core.config_store import ConfigStore
from constructor import Config
from hydra import compose, initialize
from omegaconf import OmegaConf

# Path to the root of the project
CS = ConfigStore.instance()
CS.store(name="config", node=Config)
ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_FOLDER = str(ROOT_DIR / "config")
CONFIG_NAME = "config"
initialize(config_path='../../config', job_name="config")
cfg = compose(config_name=CONFIG_NAME)