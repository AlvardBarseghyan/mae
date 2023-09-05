import os
import sys
sys.path.append('../')
import argparse
import math
import json

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

def get_embeds(model_name, img_dir, label_dir, embed_out, img_out=None, label_out=None, img_size=224, device='cuda'):
    encoder = Encoder(model_name, device=device, img_size=img_size)
    encoder.eval()
    patch_size = encoder.model_config['patch']
    #n_patches = img_size / patch_size
    img_folders = get_sub_folders(img_dir)
    print(img_out)
    for img_folder in tqdm(img_folders):
        frame_paths = get_files(img_folder, '.jpg')
        for frame_path in frame_paths:
            image = cv2.imread(frame_path)
            
            #Pad image to have shapes divisible on pitch size
            # new_height = math.ceil(image.shape[0] / patch_size) * patch_size
            # new_width = math.ceil(image.shape[1] / patch_size) * patch_size
            # black_image = np.zeros((new_height, new_width, 3))
            # black_image[:image.shape[0], :image.shape[1]] = image
            # image = black_image
            orig_shape = image.shape
            image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            
            if img_out is not None:
                # Save resized image
                out_path = frame_path.replace(img_dir, img_out)
                make_dirs(dir_name(out_path))
                cv2.imwrite(out_path, image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            proc_image = image_normalize(image)
            proc_image = torch.from_numpy(proc_image).float().unsqueeze(0)
            proc_image = torch.einsum('nhwc->nchw', proc_image)

            with torch.no_grad():
                img_enc = encoder(proc_image.to(device))
            
            img_enc_np = img_enc.detach().cpu().squeeze(0).numpy()
            
            # Save embeds
            out_path = frame_path.replace(img_dir, embed_out)
            out_path = out_path[:-4] + '.npy'
            make_dirs(dir_name(out_path))
            np.save(out_path, img_enc_np)

        if label_out is not None:
            x_scale = img_size / orig_shape[1]
            y_scale = img_size / orig_shape[0]
            name = file_name(img_folder)
            label_path = os.path.join(label_dir, f'{name}.json')
            with open(label_path) as f:
                label_data = json.load(f)

            for frame_id in range(len(label_data)):
                for obj_id in range(len(label_data[frame_id]['labels'])):
                    label_data[frame_id]['labels'][obj_id]['box2d']['x1'] *= x_scale
                    label_data[frame_id]['labels'][obj_id]['box2d']['x2'] *= x_scale
                    label_data[frame_id]['labels'][obj_id]['box2d']['y1'] *= y_scale
                    label_data[frame_id]['labels'][obj_id]['box2d']['y2'] *= y_scale
            
            # Save labels
            out_path = label_path.replace(label_dir, label_out)
            make_dirs(dir_name(out_path))
            with open(out_path, 'w') as f:
                json.dump(label_data, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="dino")
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--cuda_num', type=int, default=0)
    args = parser.parse_args()
    get_embeds(
                model_name=args.model, 
                img_dir='/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/images/track/val',
                label_dir='/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/labels/box_track_20/val',
                embed_out=f'/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/embeds_{args.model}_{args.img_size}/track/val',
                #img_out=f'/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/images_{args.img_size}/track/val',
                #label_out=f'/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/labels_{args.img_size}/box_track_20/val',
                img_size=args.img_size,
                device=f'cuda:{args.cuda_num}'
            )