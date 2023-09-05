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
from torchvision.ops import RoIAlign
from modeling import Encoder
from tracking.utils import file_utils as futils
#from roi_head import RoIHead, SimpleRoIHead



def image_normalize(image):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    image = image / 255.
    image = image - imagenet_mean
    image = image / imagenet_std
    return image


def inference_model(model, img, device='cuda'):
    img = image_normalize(img)
    patch_size = model.model_config['patch']
    emb_size = model.model_config['emb_size']
    out = torch.zeros((img.shape[0] // patch_size, img.shape[1] // patch_size, emb_size), device=device)
    for i in range(img.shape[0] // 224):
        for j in range(img.shape[1] // 224):
            crop_img = img[224*i: 224*(i+1), 224*j: 224*(j+1)]
            crop_img = torch.from_numpy(crop_img).float().unsqueeze(0)
            crop_img = torch.einsum('nhwc->nchw', crop_img)
            with torch.no_grad():
                crop_out = model(crop_img.to(device))
            #crop_out = crop_out.detach().cpu().squeeze(0).numpy()
            crop_out = crop_out.detach().squeeze(0)
            h, w = 224//patch_size, 224//patch_size
            crop_out = crop_out.reshape((h, w, -1))
            out[h*i: h*(i+1), w*j: w*(j+1)] = crop_out
    return out


def get_embeds(model_name, img_dir, label_dir, bbox_embed_out, 
               label_out=None, img_size=(224 * 3, 224 * 5), device='cuda'):
    model = Encoder(model_name, device=device)
    model.eval()
    patch_size = model.model_config['patch']
    emb_size = model.model_config['patch']

    roi_head =  RoIAlign((7,7), spatial_scale=1 / patch_size, sampling_ratio=-1)
    #SimpleRoIHead(featmap_strides=[patch_size], out_channels=emb_size, device=device)

    video_labels = sorted(futils.get_files(label_dir, '.json'))[:5]
    for video_label in tqdm(video_labels):
        with open(video_label) as f:
            label_data = json.load(f)
        for i in range(len(label_data)):
            frame_name = label_data[i]['name']
            video_name = label_data[i]['videoName']
            img_path = os.path.join(img_dir, video_name, frame_name)
            image = cv2.imread(img_path)
            
            orig_shape = image.shape
            image = cv2.resize(image, img_size[::-1], interpolation=cv2.INTER_CUBIC)
            label_data[i]['imageShape'] = img_size
            x_scale = img_size[1] / orig_shape[1]
            y_scale = img_size[0] / orig_shape[0]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out = inference_model(model, image, device)
            out = out.unsqueeze(0)
            out = torch.einsum('nhwc->nchw', out)

            for obj_id in range(len(label_data[i]['labels'])):
                bbox = label_data[i]['labels'][obj_id]['box2d']
                bbox['x1'] *= x_scale
                bbox['x2'] *= x_scale
                bbox['y1'] *= y_scale
                bbox['y2'] *= y_scale
                label_data[i]['labels'][obj_id]['box2d'] == bbox
                bbox_tensor = torch.tensor([[0, bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]], device=device)
                feat = roi_head(out, bbox_tensor)
                feat = feat.cpu().numpy()
                label_data[i]['labels'][obj_id]['features'] = feat[0]
        
        # Save bbox embeds
        out_path = video_label.replace(label_dir, bbox_embed_out)[:-5] + '.npy'
        futils.make_dirs(futils.dir_name(out_path))
        np.save(out_path, label_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="dino")
    parser.add_argument('--cuda_num', type=int, default=0)
    args = parser.parse_args()
    img_size = (224 * 3, 224 * 5)
    get_embeds(
                model_name=args.model, 
                img_dir='/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/images/track/val',
                label_dir='/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/labels/box_track_20/val',
                bbox_embed_out=f'/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/bbox_embeds_{args.model}_{img_size[0]}_{img_size[1]}_new/track/val',
                #img_out=f'/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/images_{args.img_size}/track/val',
                #label_out=f'/mnt/lwll/lwll-coral/hrant/mae_checkpoints/tracking/bdd100k/labels_{args.img_size}/box_track_20/val',
                img_size = img_size,
                device=f'cuda:{args.cuda_num}'
            )