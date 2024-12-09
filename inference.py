import os
import glob

import warnings

from random import randint
from argparse import ArgumentParser

import torchvision.transforms.functional as tf

warnings.filterwarnings(action='ignore')


import numpy as np

from PIL import Image
from tqdm import tqdm
from scipy.interpolate import griddata as interp_grid
from scipy.ndimage import minimum_filter, maximum_filter

import torch
import torch.nn.functional as F
from mv_diffusion import mvdream_diffusion_model
from mv_diffusion_SR import mvdream_diffusion_model as mvdream_diffusion_model_SR
from transformers import CLIPTextModel, CLIPTokenizer
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import re
from torch.autograd import Variable
from math import exp
import argparse
import gc

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--single_view', action='store_true', help='Use single view mode')
parser.add_argument('--super_resolution', action='store_true', help='Enable super resolution')
parser.add_argument('--base_model_path', type=str, help='Base model directory path')
parser.add_argument('--source_imgs_dir', type=str, help='Source images directory path')
parser.add_argument('--warp_root_dir', type=str, help='Warp images directory path')
parser.add_argument('--output_dir', type=str, help='Output directory path')

# Parse arguments
args = parser.parse_args()

def PIL2tensor(height,width,num_frames,masks,warps,logicalNot=False):
    channels = 3
    pixel_values = torch.empty((num_frames, channels, height, width))
    condition_pixel_values = torch.empty((num_frames, channels, height, width))
    masks_pixel_values = torch.ones((num_frames, 1, height, width))
    
    # input_ids
    prompt = ''

    for i, img in enumerate(masks):
        img = masks[i]
        img = img.convert('L') # make sure channel 1
        img_resized = img.resize((width, height)) # hard code here
        img_tensor = torch.from_numpy(np.array(img_resized)).float()

        # Normalize the image by scaling pixel values to [0, 1]
        img_normalized = img_tensor / 255
        mask_condition = (img_normalized > 0.9).float()
        
        masks_pixel_values[i] = mask_condition
    
    for i, img in enumerate(warps):
        # Resize the image and convert it to a tensor
        img_resized = img.resize((width, height)) # hard code here
        img_tensor = torch.from_numpy(np.array(img_resized)).float()

        # Normalize the image by scaling pixel values to [-1, 1]
        img_normalized = img_tensor / 127.5 - 1

        img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images

        if(logicalNot):
            img_normalized = torch.logical_not(masks_pixel_values[i])*(-1) + masks_pixel_values[i]*img_normalized
        condition_pixel_values[i] = img_normalized
        
    return [prompt], {
            'conditioning_pixel_values': condition_pixel_values, # [-1,1]
            'masks': masks_pixel_values# [0,1]
            }
    
def get_image_files(folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    image_names = [os.path.basename(file) for file in image_files]
    
    return image_names

single_view = args.single_view
super_resolution = args.super_resolution
base_model_path = args.base_model_path
if(single_view):
    mv_unet_path = base_model_path + "/unet/single/ema-checkpoint"
    print(mv_unet_path)
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
else:
    mv_unet_path = base_model_path + "/unet/sparse/ema-checkpoint"
    print(mv_unet_path)
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
rgb_model = mvdream_diffusion_model(base_model_path,mv_unet_path,tokenizer,seed=12345)
mv_unet_path = base_model_path + "/unet/SR/ema-checkpoint"
rgb_model_SR = mvdream_diffusion_model_SR(base_model_path,mv_unet_path,tokenizer,quantization=False,seed=12345)
        

source_imgs_dir = args.source_imgs_dir
warp_root_dir = args.warp_root_dir
output_root_dir = args.output_dir
os.makedirs(output_root_dir, exist_ok=True)


height_mvd = 512
width_mvd = 512
masks_infer = []
warps_infer = []
input_names = []

gt_num_b = 0
mask2 = np.ones((height_mvd,width_mvd), dtype=np.float32)

image_names_ref = get_image_files(source_imgs_dir)
fimage = Image.open(os.path.join(source_imgs_dir + image_names_ref[0]))
(width, height)= fimage.size
for imn in image_names_ref:
    masks_infer.append(Image.fromarray(np.repeat(np.expand_dims(np.round(mask2*255.).astype(np.uint8),axis=2),3,axis=2)).resize((width_mvd, height_mvd)))
    warps_infer.append(Image.open(os.path.join(source_imgs_dir + imn)))
    input_names.append(imn)
    gt_num_b = gt_num_b + 1



image_files = glob.glob(os.path.join(warp_root_dir, "warp_*"))
image_names = [os.path.basename(image) for image in image_files]

image_names.sort()
print(image_names)
for ins in image_names:
    warps_infer.append(Image.open(os.path.join(warp_root_dir, ins))) 
    masks_infer.append(Image.open(os.path.join(warp_root_dir, ins.replace('warp','mask'))))
    input_names.append(ins)



print('sequence length:', len(warps_infer))
images_predict = []
images_mask_p = []
images_predict_names = []

grounp_size = len(masks_infer)
for i in range(0, len(masks_infer[gt_num_b:]), grounp_size):
    if(len(images_predict)!=0):
        masks_infer_batch = masks_infer[:gt_num_b] + [masks_infer_batch[-1]] + masks_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
        warp_infer_batch = warps_infer[:gt_num_b] + [images_predict[-1]] + warps_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
        input_names_batch = input_names[:gt_num_b] + [input_names_batch[len(masks_infer_batch)//2]] + [input_names_batch[-1]] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]
    else:
        masks_infer_batch = masks_infer[:gt_num_b] + masks_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
        warp_infer_batch = warps_infer[:gt_num_b] + warps_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
        input_names_batch = input_names[:gt_num_b] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]

    prompt, batch = PIL2tensor(height_mvd,width_mvd,len(masks_infer_batch),masks_infer_batch,warp_infer_batch,logicalNot=False)
    if(len(images_predict)!=0):
        images_predict_batch = rgb_model.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd,width_mvd,gt_num_frames=gt_num_b,output_type='pil')
        for jj in range(gt_num_b+1,len(images_predict_batch)):
            images_predict.append(images_predict_batch[jj])
            images_mask_p.append(batch['masks'][0][jj][0].cpu().numpy())
            images_predict_names.append(input_names_batch[jj])
    else:
        images_predict_batch = rgb_model.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd,width_mvd,gt_num_frames=gt_num_b,output_type='pil')
        for jj in range(gt_num_b,len(images_predict_batch)):
            images_predict.append(images_predict_batch[jj])
            images_mask_p.append(batch['masks'][0][jj][0].cpu().numpy())
            images_predict_names.append(input_names_batch[jj])
        
for jj in range(len(images_predict)):
    images_predict[jj].resize((width, height)).save(os.path.join(output_root_dir,"predict_{}.jpg".format(images_predict_names[jj])))

if(super_resolution):
    del mvdream_diffusion_model
    gc.collect()
    torch.cuda.empty_cache()

    masks_infer_SR = []
    warps_infer_SR = []
    mask2 = np.ones((height_mvd*2,width_mvd*2), dtype=np.float32)

    for imn in image_names_ref:
        masks_infer_SR.append(Image.fromarray(np.repeat(np.expand_dims(np.round(mask2*255.).astype(np.uint8),axis=2),3,axis=2)).resize((width_mvd, height_mvd)))
        warps_infer_SR.append(Image.open(os.path.join(source_imgs_dir + imn)))

    for i in range(len(images_predict)):
        masks_infer_SR.append(masks_infer[i])
        warps_infer_SR.append(images_predict[i])

    images_predict = []
    images_predict_names = []
    # grounp_size = min((len(masks_infer_SR) + 5)//2,50)
    grounp_size = (len(masks_infer_SR) + 3) // 2
    # grounp_size = (len(masks_infer_SR) + 3)
    print('grounp_size:',grounp_size)
    for i in range(0, len(masks_infer_SR[gt_num_b:]), grounp_size):
        if(len(images_predict)!=0):
            masks_infer_batch = masks_infer_SR[:gt_num_b] + [masks_infer_batch[len(masks_infer_batch)//2]] + [masks_infer_batch[-1]] + masks_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
            warp_infer_batch = warps_infer_SR[:gt_num_b] + [images_predict[len(images_predict)//2]] + [images_predict[-1]] + warps_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
            input_names_batch = input_names[:gt_num_b] + [input_names_batch[len(masks_infer_batch)//2]] + [input_names_batch[-1]] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]
        else:
            masks_infer_batch = masks_infer_SR[:gt_num_b] + masks_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
            warp_infer_batch = warps_infer_SR[:gt_num_b] + warps_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
            input_names_batch = input_names[:gt_num_b] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]

        
        prompt, batch = PIL2tensor(height_mvd*2,width_mvd*2,len(masks_infer_batch),masks_infer_batch,warp_infer_batch)
        if(len(images_predict)!=0):
            images_predict_batch = rgb_model_SR.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd*2,width_mvd*2,gt_num_frames=gt_num_b,output_type='pil')
            for jj in range(gt_num_b+2,len(images_predict_batch)):
                images_predict.append(images_predict_batch[jj])
                images_predict_names.append(input_names_batch[jj])
        else:
            images_predict_batch = rgb_model_SR.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd*2,width_mvd*2,gt_num_frames=gt_num_b,output_type='pil')
            for jj in range(gt_num_b,len(images_predict_batch)):
                images_predict.append(images_predict_batch[jj])
                images_predict_names.append(input_names_batch[jj])
        gc.collect()
        torch.cuda.empty_cache()


    for jj in range(len(images_predict)):
        images_predict[jj].resize((width, height)).save(os.path.join(output_root_dir,"SR_predict_{}.jpg".format(images_predict_names[jj])))
    
