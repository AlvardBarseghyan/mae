import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics import AUROC
from binary_classifier import Classifier, Model
from torchmetrics.classification import BinaryF1Score

with open('preprocessing/labels.txt') as f:
    labels = {
        int(line[22:25].strip()):
        (line[:22].strip().replace("'", ""), 
         int(line[25:].strip()))
        for line in f.readlines()
    }

eval_labels = [i for i in labels if 0 <= labels[i][1] < 255]
eval_label_names = [labels[i][0] for i in labels if 0 <= labels[i][1] < 255]
t = {k:v for k, v in zip(eval_labels, eval_label_names)}


path = '/home/hkhachatrian/mae/checkpoints'

auc = AUROC(task="binary")
ckpts = [ f'dino_{t[v]}_patches_1000/epoch=299.ckpt'.replace(' ', '_') for v in eval_labels]

classes = [v for v in eval_labels]

print('Reading embeddings')
embeds_val = np.load('/mnt/lwll/lwll-coral/hrant/cs_patches_256/dino_embeds_val.npy', allow_pickle=True)
embeds_val = torch.tensor(embeds_val)


print('Reading model')

res = {}

for ckpt, cls in zip(ckpts, classes):
    # print(os.path.join(path, ckpt))
    model = Model(768, num_classes=1)
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
    labels_val = np.load('/mnt/lwll/lwll-coral/hrant/cs_patches_256/dino_labels_val.npy', allow_pickle=True)
    labels_val[labels_val!=cls] = 0
    labels_val[labels_val==cls] = 1

    roc = auc(torch.tensor(predictions_binary), torch.tensor(labels_val))
    res[ckpt] = (cls, roc.item())
    print(ckpt, roc, cls)


with open('./dino_1000_299_eval.json', 'w') as f:
    json.dump(res, f)