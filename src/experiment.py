import uuid
import logger
import yaml
import os
import torch

import pandas as pd

from tqdm import tqdm
from typing import List, Optional, Tuple, Union
from diffusers import pipelines
from datasets import load_dataset
from ssim_idea import SLDPipeline
from utils.experiment_utils.utils import create_dirs, get_model_config


LOGGER = logger.get_logger(__name__)

RESULTS_CONFIG_COL_NAMES = {
    'image_id',
    'image',
    'original_prompt_id',
    'original_prompt',
    'inappropriate_percentage',
    'sg_inappropriate_percentage',
    'nudity_percentage',
    'sg_nudity_percentage',
    'q16_percentage',
    'sg_q16_percentage',
    'sd_safety_percentage',
    'sd_seed',
    'sd_guidance_scale',
    'sd_image_width',
    'sd_image_height',
    'model_type'
}


def generate_images_from_models(models_config_path: str):

    models_config = get_model_config(models_config_path)
    LOGGER.info(f"Models config: {models_config}")
    LOGGER.info(f"Got models: {[model['name']+', ' for model in models_config['models']]}")
    
    for model_config in models_config['models']:
        LOGGER.info(f"Processing model {model_config['name']}")
        model_name = model_config['name']
        model_parameters = model_config['parameters']
        batch_size = model_config['batch_size']
        data_path = model_config['data_path']
        dataset, relevant_dataset_cols = load_dataset(data_path)
        model, gen = load_model()
        
        image_dir, results_dir, config_dir = create_dirs(model_name)

        save_model_config(model_config, config_dir)
        last_index = get_or_create_last_index(config_dir)

        LOGGER.info(f"Starting inference for model {model_name}")
        for i in tqdm(range(last_index, len(dataset), batch_size)):
            LOGGER.info(f"Processing batch {i} to {i+batch_size}")
            batch = get_batch(batch_size, dataset, i)

            results = model_inference(model, gen, batch, model_parameters, relevant_dataset_cols)
            results_df = pd.DataFrame(results)
            
            LOGGER.info(f"Saving images and results for batch {i} to {i+batch_size}")
            results_df = save_images(image_dir, results_df)
            save_results(results_dir, i, results_df)
            update_last_index(batch_size, config_dir, i)
            LOGGER.info(f"Finished batch {i} to {i+batch_size}")


def load_dataset(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    i2p_ds = pd.read_csv(data_path)
    relevant_dataset_cols = [col for col in i2p_ds.columns if col in RESULTS_CONFIG_COL_NAMES]
    return i2p_ds, relevant_dataset_cols

def load_model() -> Tuple[SLDPipeline, torch.Generator]:
    stable_diffusion_src_path = "CompVis/stable-diffusion-v1-4"
    device='cuda'

    pipe = SLDPipeline.from_pretrained(
        stable_diffusion_src_path,
        safety_checker = None,
    ).to(device)
    gen = torch.Generator(device=device)
    return pipe, gen

def save_model_config(model_config: dict, config_dir: str):
    LOGGER.info(f"Saving model config to {config_dir}")
    with open(f"{config_dir}/model_config.yaml", 'w') as file:
        yaml.dump(model_config, file)

def get_or_create_last_index(config_dir: str) -> int:
    try:
        with open(f"{config_dir}/last_index.txt", 'r') as file:
            last_index = int(file.read())
    except FileNotFoundError:
        last_index = 0
    LOGGER.info(f"Last index is {last_index}")
    return last_index

def get_batch(batch_size: int, dataset: pd.DataFrame, i: int) -> pd.DataFrame:
    if i + batch_size > len(dataset):
        batch = dataset.iloc[i:]
    else:
        batch = dataset.iloc[i:i+batch_size]
    return batch

def model_inference(model: SLDPipeline, gen:torch.Generator, dataset_batch: pd.DataFrame, model_parameters: dict, relevant_dataset_cols: List[str]) -> List[dict]:
    results = []
    for row in dataset_batch.iterrows():
        row = row[1]
        prompts = [row['prompt']] * 10
        original_prompt_id = uuid.uuid4()
        seed = row['seed']
        outputs = generate_images_from_model(prompts, model, gen, seed, **model_parameters)
        images, _ = outputs

        for image in images:
            result = {col: None for col in RESULTS_CONFIG_COL_NAMES}
            result['image_id'] = str(uuid.uuid4())
            result['original_prompt_id'] = original_prompt_id
            result['original_prompt'] = row['prompt']
            result['image'] = image

            # Copy relevant columns from dataset_batch
            for col in relevant_dataset_cols:
                result[col] = row[col]

            results.append(result)

    return results     

def generate_images_from_model(prompts: Union[str, List[str]], model: SLDPipeline, gen:torch.Generator,  seed: int, **model_parameters) -> List[pipelines.stable_diffusion.StableDiffusionPipelineOutput]:
    model.manual_seed(seed)

    return model(prompt=prompts, generator=gen,
                **model_parameters)
            #    guidance_scale=scale,
            #    sld_warmup_steps=7,
            #    sld_guidance_scale=2000,
            #    sld_threshold=0.025,
            #    sld_momentum_scale=0.5,
            #    sld_mom_beta=0.7

def save_images(image_dir: str, results_df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info(f"Saving images to {image_dir}")
    for row in results_df.iterrows():
        row = row[1]
        row['image'].save(f"{image_dir}/{row['image_id']}.png")
    results_df.drop(columns='image', inplace=True)
    return results_df

def save_results(results_dir: str, i: int, results_df: pd.DataFrame):
    LOGGER.info(f"Saving results to {results_dir}")
    if i == 0:
        results_df.to_csv(f"{results_dir}/results.csv", index=False)
    else:
        results_df.to_csv(f"{results_dir}/results.csv", mode='a', header=False, index=False)

def update_last_index(batch_size: int, config_dir: str, i: int):
    LOGGER.info(f"Updating last index to {i+batch_size}")
    with open(f"{config_dir}/last_index.txt", 'w') as file:
        file.write(str(i+batch_size))












