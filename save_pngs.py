import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--ann_root', default='/mnt/lwll/lwll-coral/hrant/annotations/')
parser.add_argument('--annotation', default='cs4pc_200_train.npy', type=str)
parser.add_argument('--saving_path')
# parser.add_argument('--complete_npy', default=name, type=str)
# parser.add_argument('--save_name_npy', default='cs4pc_upsampled_200_train.npy')
args = parser.parse_args()

annots = np.load(os.path.join(args.ann_root, args.annotation), allow_pickle=True)

os.makedirs(args.saving_path, exist_ok=True)
for ann in tqdm(annots, total=len(annots)):
    file_name = ann['file_name'].split('/')[-1].strip()
    path = os.path.join(args.saving_path, file_name)
    cv2.imwrite(path, ann['patch_labels'])


