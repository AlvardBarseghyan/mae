import os
import sys
sys.path.append('../')
import argparse
import math

import numpy as np
import cv2
from tqdm import tqdm
import torch

from modeling import Encoder
from tracking.utils.file_utils import *



def image_normalize(image):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    image = image / 255.
    image = image - imagenet_mean
    image = image / imagenet_std
    return image

def get_embeds(model_name, data_dir, out_dir, device='cuda'):
    encoder = Encoder(model_name, device=device)
    encoder.eval()
    img_paths = get_files(data_dir, '.jpg')
    for img_path in tqdm(img_paths):

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        patch_size = encoder.model_config['patch']

        #Pad image to have shapes divisible on pitch size
        new_height = math.ceil(image.shape[0] / patch_size) * patch_size
        new_width = math.ceil(image.shape[1] / patch_size) * patch_size
        black_image = np.zeros((new_height, new_width, 3))
        black_image[:image.shape[0], :image.shape[1]] = image
        image = black_image

        proc_image = image_normalize(image)
        proc_image = torch.from_numpy(proc_image).float().unsqueeze(0)
        proc_image = torch.einsum('nhwc->nchw', proc_image)

        with torch.no_grad():
            img_enc = encoder(proc_image.to(device))
        
        img_enc_np = img_enc.detach().cpu().squeeze(0).numpy()
        
        out_path = img_path.replace(data_dir, out_dir)
        out_path = out_path[:-4] + '.npy'
        make_dirs(dir_name(out_path))
        
        np.save(out_path, img_enc_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="dino")
    args = parser.parse_args()
    get_embeds(
                model_name=args.model, 
                data_dir='/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/images/track/val',
                out_dir=f'/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/embeds/{args.model}/track/val',
                device='cuda:1'
            )