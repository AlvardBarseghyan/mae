import torch
import os
import numpy as np
from tqdm.notebook import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt
import h5py

from sklearn.mixture import GaussianMixture


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

np.random.seed(32635)

model = 'mae'
num_classes = 33
pathTrain = f'/mnt/lwll/lwll-coral/hrant/embeddings/cs_patches_256/{model}_embeds_train_72.npy'
pathVal = f'/mnt/lwll/lwll-coral/hrant/embeddings/cs_patches_256/{model}_embeds_val_30.npy'

pathLabelsTrain = f'/mnt/lwll/lwll-coral/hrant/embeddings/cs_patches_256/{model}_labels_train_72.npy'
pathLabelsVal = f'/mnt/lwll/lwll-coral/hrant/embeddings/cs_patches_256/{model}_labels_val_30.npy'

XTrain_orig = np.load(pathTrain)
XValidation_orig = np.load(pathVal)

yTrain = np.load(pathLabelsTrain)
yVal = np.load(pathLabelsVal) 

XTrain = XTrain_orig
XValidation = XValidation_orig

classes = np.unique(yTrain)
gmms = {}
for cls in tqdm(classes, total=len(classes)):
    n_comp = 15 if cls == 0 else 3
    gmms[cls] = GaussianMixture(n_components=n_comp, random_state=0).fit(XTrain[yTrain==classes[cls]])


for i, gm in enumerate(gmms.values()):
    if not gm.converged_:
        print('class id:', i, 'not converged', gm.converged_)


np.save(f'/mnt/lwll/lwll-coral/hrant/embeddings/cs_patches_256/gmms/{model}_gmms_train_72.npy', gmms)
