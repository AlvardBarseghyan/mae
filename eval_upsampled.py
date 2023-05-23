import os
import sys
import cv2
sys.path.append('..')

import numpy as np
import torch
import models_mae
# from knn_test import KNN
from collections import OrderedDict

from pl_train import Encoder, LightningMAE, get_dino_model, get_model_sup_vit

from torch import nn
from tqdm import tqdm
from util.get_dataloader import get_dataloader
import argparse

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--checkpoint', default='', help='absolute path to checkpoint to be loaded', )
parser.add_argument('--saving_path')
parser.add_argument('--model')
parser.add_argument('--conv_out_size', default=1024, type=int)

args = parser.parse_args()

activation = nn.Softmax(dim=-1)

annotation_train, annotation_val = './annotations/cs4pc_256_train.npy', './annotations/cs4pc_256_val.npy'
# annotation_train, annotation_val = './annotations/cs_train.npy', './annotations/cs_val.npy'
dataset_name = 'cs'

annots = np.load(annotation_val, allow_pickle=True).item()
print('!!!!!!!!!!!!!',  annots['images'][0]['patch_labels'].shape)
dataloader, dataloader_val, num_classes, dataset = get_dataloader(dataset_name=dataset_name, \
    train_annotation_file=annotation_train, val_annotation_file=annotation_val, \
    intersection_threshold=0.3, batch_size=20, \
    weighted=False, return_dataset=True, resize_image=True)


def get_model(model_name):
    device = 'cuda'
    if model_name == 'mae':
        arch='mae_vit_large_patch16'
        model = getattr(models_mae, arch)()

        chkpt_dir = './mae_visualize_vit_large_ganloss.pth'
        checkpoint = torch.load(chkpt_dir, map_location=device)
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        emb_size = 1024

    if model_name == 'dino':
        model = get_dino_model()
        emb_size = 768

    if model_name == 'sup_vit':
        model = get_model_sup_vit()
        emb_size = 1024

    encoder = Encoder(model=model, num_classes=33, backbone_freeze=True,
                      model_name=model_name, emb_size=emb_size, deconvolution=True,
                      bilinear_upsampling=True, out_size=args.conv_out_size)
    return encoder.to('cuda')

model = get_model(args.model)

chkp = torch.load(args.checkpoint)

keys = [i for i in model.state_dict().keys()]

od = OrderedDict()

for key_chpt, key_model in zip(chkp['state_dict'].keys(), keys):
    if key_model == key_chpt[10:]:
        od[key_model] = chkp['state_dict'][key_chpt]
    else:
        print(key_model, key_chpt, sep='\t')

model.load_state_dict(od, strict=False)

model.eval()

# ebeds_val = []
predictions_val_up_mae = []
images = []
labels_val = []
for ds in tqdm(dataloader_val, total=len(dataloader_val)):
    img = torch.einsum('nhwc->nchw', ds['image'])
    images.extend(ds['file_name'])
#     print(img.shape)
    img_enc = model(img.float().to('cuda'))
#     ebeds_val.append(img_enc.to('cpu'))
    labels_val.append(ds['patch_56x56'])
    # labels_val.append(ds['patch_dino'])
    pred = activation(img_enc).argmax(dim=-1).to('cpu').detach().numpy()
    # pred = cv2.resize(pred, (2048, 1024), interpolation=cv2.INTER_NEAREST_EXACT)
    predictions_val_up_mae.append(torch.tensor(pred))

predictions_val_up_mae = torch.cat(predictions_val_up_mae, dim=0)

annots = np.load(annotation_val, allow_pickle=True).item()
print("Shapes: annots, preds", len(annots['images']), annots['images'][0]['patch_labels'].shape, predictions_val_up_mae.shape)
for i, preds in enumerate(predictions_val_up_mae):
    annots['images'][i]['patch_labels_gt'] = annots['images'][i]['patch_labels']
    annots['images'][i]['patch_labels'] = preds.detach().numpy()

np.save(args.saving_path, annots)
