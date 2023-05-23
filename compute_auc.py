import os
import sys

sys.path.append('..')

import numpy as np
import torch
import models_mae
from knn_test import KNN
from collections import OrderedDict

from pl_train import Encoder, LightningMAE, get_dino_model

from torch import nn
from tqdm import tqdm
from util.get_dataloader import get_dataloader

from torchmetrics import AUROC

activation = nn.Softmax(dim=-1)
auc = AUROC(task="multiclass", num_classes=34)

annotation_train, annotation_val = './annotations/cs4pc_256_train.npy', './annotations/cs4pc_256_val.npy'
dataset_name = 'cs'


dataloader, dataloader_val, num_classes, dataset = get_dataloader(dataset_name=dataset_name, \
    train_annotation_file=annotation_train, val_annotation_file=annotation_val, \
    intersection_threshold=0.3, batch_size=20, \
    weighted=False, return_dataset=True)

def get_model(model_name, emb_size):
    device = 'cuda'
    arch='mae_vit_large_patch16'
    model_mae = getattr(models_mae, arch)()

    chkpt_dir = './mae_visualize_vit_large_ganloss.pth'
    checkpoint = torch.load(chkpt_dir, map_location=device)
    msg = model_mae.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    encoder = Encoder(model=model_mae, num_classes=33, backbone_freeze=True,
                      model_name=model_name, emb_size=emb_size, deconvolution=False,
                      bilinear_upsampling=True)
    return encoder.to('cuda')

model = get_model('mae', 1024)


def f(x):
    try: return int(x[6:9])
    except ValueError:
        try: return int(x[6:8])
        except ValueError: return int(x[6:7])


def load_checkpoints(model, weights):
    keys = [i for i in model.state_dict().keys()]

    od = OrderedDict()

    for key_chpt, key_model in zip(weights['state_dict'].keys(), keys):
        if key_model == key_chpt[10:]:
            od[key_model] = weights['state_dict'][key_chpt]
        else:
            print(key_model, key_chpt, dep='\t')

    model.load_state_dict(od, strict=False)
    return model


root = '/mnt/lwll/lwll-coral/hrant/mae_checkpoints/'
paths = [os.path.join(root, f'cs_256_mae_bilinear_upsampling_lr_1e{i}') for i in [2, 3, 4]]
res = []
for path in paths:
    tmp_dct = {}
    tmp = os.listdir(path)
    tmp.remove('last.ckpt')
    checkpoints = sorted(tmp, key=f)
    for ckpt in tqdm(checkpoints, total=len(checkpoints)):
        ckpt_path = os.path.join(path, ckpt)
        weights = torch.load(ckpt_path)
        model = load_checkpoints(model, weights)

        model.eval()
        predictions = []
        labels = []
        for ds in dataloader:
            img = torch.einsum('nhwc->nchw', ds['image']).to('cuda')
            img_enc = model(img.float())
            img = img.detach().to('cpu')
            img_enc = img_enc.detach().to('cpu')
            predictions.append(activation(img_enc))
            labels.append(ds['patch_56x56'])

        labels = torch.cat(labels, dim=0).reshape(-1)
        predictions = torch.cat(predictions, dim=0).reshape(-1, predictions[0].shape[-1])

        tmp_dct[f(ckpt)] = auc(predictions, labels)
    res.append(tmp_dct)


np.save('./auc_preds_train.npy', res)

