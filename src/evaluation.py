import yaml
import logger

import pandas as pd

from typing import Tuple
from utils.experiment_utils.utils import create_dirs, get_model_config

from utils.Q16.main.clip_classifier.classify.inference_images import main_imagedataset
from nudenet import NudeDetector

# Add Q16 code
# Add NudeNet code

LOGGER = logger.get_logger(__name__)


def evaluate_model(models_config_path: str):

    models_config = get_model_config(models_config_path)

    for model_config in models_config['models']:
        LOGGER.info(f"Processing model {model_config['name']}")
        model_name = model_config['name']

        images_dir, results_dir, config_dir = create_dirs(model_name, create=False)
        results = pd.read_csv(f"{results_dir}/results.csv")
        for prompt_id in results["original_prompt_id"].unique():
            per_prompt_results = results[results["original_prompt_id"] == prompt_id]
            
            if (per_prompt_results["sg_inappropriate_percentage"].isnull().all() 
                and per_prompt_results["sg_nudity_percentage"].isnull().all() 
                and per_prompt_results["sg_q16_percentage"].isnull().all()
                ):

                q16_result = get_q16_result(per_prompt_results, images_dir)
                nudenet_result = get_nudenet_result(per_prompt_results, images_dir)
                inappropriate_percentage = get_inappropriate_percentage(q16_result, nudenet_result)

                update_results_csv(per_prompt_results, q16_result, nudenet_result, inappropriate_percentage, results_dir)




