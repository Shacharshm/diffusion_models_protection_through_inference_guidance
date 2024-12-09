import os
import logging
import yaml

from typing import Tuple

LOGGER = logging.getLogger(__name__)

def create_dirs(model_name: str, create: bool=True) -> Tuple[str, str, str]:
    LOGGER.info(f"Creating directories for model {model_name}")
    images_dir = f"result_images/{model_name}/images"
    results_dir = f"result_images/{model_name}/results"
    config_dir = f"result_images/{model_name}/config"
    if create:
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)

    return images_dir, results_dir, config_dir

def get_model_config(models_config_path: str) -> dict:
    with open(models_config_path, 'r') as file:
        models_config = yaml.safe_load(file)
    return models_config