import sys
from diffusers import (
    UniPCMultistepScheduler,
    DDIMScheduler
)
from pipeline_mvd_warp_mix_classifier_SR import MVDreamPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from mv_unet import MultiViewUNetModel
from accelerate.utils import ProjectConfiguration, set_seed
import argparse
import os
import sys
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection



class mvdream_diffusion_model:
    def __init__(self, base_model_path,mv_unet_path,tokenizer,quantization=False,seed=12345):

        generator = torch.Generator("cuda").manual_seed(seed)

        unet = MultiViewUNetModel.from_pretrained(
        mv_unet_path,
        torch_dtype=torch.float16
        )
        feature_extractor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(base_model_path + "/CLIP-ViT-H-14-laion2B-s32B-b79K")
        image_encoder: CLIPVisionModelWithProjection = CLIPVisionModelWithProjection.from_pretrained(base_model_path + "/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.pipe = MVDreamPipeline.from_pretrained(
            base_model_path, # remote weights
            unet=unet,
            torch_dtype=torch.float16,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, timestep_scaling="trailing", rescale_betas_zero_snr=True)
        
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()




    def inference_next_frame(self, prompt, batch, num_frames, height, width, gt_num_frames=1, output_type='pil'):
        batch['conditioning_pixel_values'] = torch.unsqueeze(batch['conditioning_pixel_values'], 0)
        batch['masks'] = torch.unsqueeze(batch['masks'], 0)
        
        image, image_warp = self.pipe(
            prompt=prompt,
            image=batch['conditioning_pixel_values'],
            masks=batch['masks'],
            height=height,
            width=width,
            guidance_scale=1.0,
            num_inference_steps=30, 
            guidance_rescale=0.0, 
            elevation=0,
            num_frames=batch['conditioning_pixel_values'].shape[1],
            condition_num_frames=gt_num_frames,
            gt_frame=None,  
            output_type=output_type
        )
        
        return image

