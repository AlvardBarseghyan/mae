import os
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics import AUROC
from binary_classifier import Classifier, Model
from torchmetrics.classification import BinaryF1Score


path = '/home/hkhachatrian/mae/checkpoints'

auc = AUROC(task="binary")

ckpts = ['building_patches_1000/epoch=299.ckpt', 'wall_patches_100/epoch=299.ckpt',\
 'wall_patches_1000/epoch=299.ckpt', 'wall_patches_10000/epoch=299.ckpt']
classes = [11, 12, 12, 12]

print('Reading embeddings')
embeds_val = np.load('/mnt/lwll/lwll-coral/hrant/cs_patches_256/embeds_val.npy', allow_pickle=True)
embeds_val = torch.tensor(embeds_val)


print('Reading model')


for ckpt, cls in zip(ckpts, classes):
    # print(os.path.join(path, ckpt))
    model = Model(1024, num_classes=1)
    model = Classifier.load_from_checkpoint(os.path.join(path, ckpt), model=model)
    model.to('cpu')
    model.eval()

    predictions = []
    gt_labels = []
    print('starting predictions')
    for embed in tqdm(embeds_val, total = len(embeds_val)):
        preds = model.model(embed)
        predictions.append(preds.detach())

    predictions_binary = [x.argmax().detach().numpy().item() for x in predictions]
    labels_val = np.load('/mnt/lwll/lwll-coral/hrant/cs_patches_256/labels.npy', allow_pickle=True)
    labels_val[labels_val!=cls] = 0
    labels_val[labels_val==cls] = 1

    res = auc(torch.tensor(predictions_binary), torch.tensor(labels_val))
    print(ckpt, res, cls)