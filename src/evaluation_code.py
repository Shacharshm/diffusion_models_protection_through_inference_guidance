from typing import List, Union

import pandas as pd
from utils.Q16.main.clip_classifier.classify.inference_images import main_imagedataset
from nudenet import NudeDetector
import yaml
import os
from ssim_idea import SLDPipeline
import torch
from diffusers import pipelines
from datasets import load_dataset
# Add dataset code
# Add Q16 code
# Add NudeNet code


def load_model():
    stable_diffusion_src_path = "CompVis/stable-diffusion-v1-4"
    device='cuda'

    pipe = SLDPipeline.from_pretrained(
        stable_diffusion_src_path,
        #safety_checker = None,
    ).to(device)
    gen = torch.Generator(device=device)
    return pipe, gen

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

def load_dataset():
    i2p_ds = pd.read_csv("data/i2p/i2p_benchmark.csv")
    return i2p_ds

def generate_model_parameters_url(model_parameters: dict) -> str:
    return "_".join([f"{key}_{value}" for key, value in model_parameters.items()])

def generate_images_from_models(i2p_dataset, models_config_path):
    """
    Generates images from the I2P dataset using multiple models specified in the models_config.yaml file.
    
    Parameters:
    i2p_dataset: The I2P dataset.
    models_config_path: Path to the models_config.yaml file.
    """
    with open(models_config_path, 'r') as file:
        models_config = yaml.safe_load(file)
    
    for model_config in models_config['models']:
        model_name = model_config['name']
        model_parameters = model_config['parameters']
        model = load_model()
        
        output_dir = f"result_images/{model_name}/parameters{generate_model_parameters_url(model_parameters)}"
        os.makedirs(output_dir, exist_ok=True)



