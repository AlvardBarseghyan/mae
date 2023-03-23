import torch as th
import os
import numpy as np
from tqdm.notebook import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt

np.random.seed(32635)

def cosine_distance_torch(x1, x2=None):
    x2 = x1 if x2 is None else x2
    # w1 = x1.norm(p=2, dim=1, keepdim=True)
    # w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return th.mm(x1, x2.t())


class_dct = {   7: ('road', 6111042),
                8: ('sidewalk', 1016635),
                11: ('building', 3822901),
                12: ('wall', 109501),
                13: ('fence', 146414),
                17: ('pole', 203783),
                19: ('traffic light', 34697),
                20: ('traffic sign', 91885),
                21: ('vegetation', 2665158),
                22: ('terrain', 193395),
                23: ('sky', 680522),
                24: ('person', 202630),
                25: ('rider', 22368),
                26: ('car', 1165026),
                27: ('truck', 44584),
                28: ('bus', 38923),
                31: ('train', 38767),
                32: ('motorcycle', 16403),
                33: ('bicycle', 69008)}

device='cpu'

def randomize_tensor(tensor):
    return tensor[th.randperm(len(tensor))]

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    # x = x.unsqueeze(1).expand(n, m, d)
    # y = y.unsqueeze(0).expand(n, m, d)

    # dist = th.linalg.vector_norm(x - y, p, 2) if th.__version__ >= '1.7.0' else th.pow(x - y, p).sum(2)**(1/p)
    dist = cosine_distance_torch(x, y)
    return dist


print("Loading...")
model = 'dino'
path_to_read = '/mnt/lwll/lwll-coral/hrant/cs_patches_256/'
# dino_embeds_train = th.from_numpy(np.load(path_to_read + f'{model}_embeds_train.npy')).to(device=device)
# dino_labels_train = th.from_numpy(np.load(path_to_read + f'{model}_labels_train.npy')).to(device=device, dtype=th.int64)
dino_embeds_train_path = path_to_read + f'{model}_embeds_train.npy'
dino_labels_train_path = path_to_read + f'{model}_labels_train.npy'
# dino_embeds_val = th.from_numpy(np.load(path_to_read + f'{model}_embeds_test.npy')).to(device=device)
dino_embeds_val_path =  path_to_read + f'{model}_embeds_test.npy'
# dino_labels_val = th.from_numpy(np.load(path_to_read + f'{model}_labels_test.npy')).to(device=device, dtype=th.int64)


with open('preprocessing/labels.txt') as f:
    labels = {
        int(line[22:25].strip()):
        (line[:22].strip().replace("'", ""),
         int(line[25:].strip()))
        for line in f.readlines()
    }

eval_labels = [i for i in labels if 0 <= labels[i][1] < 255]

eval_label_names = [labels[i][0] for i in labels if 0 <= labels[i][1] < 255]


class NN():

    def __init__(self, X = None, Y = None, p = 2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = th.from_numpy(np.load(X)).to(device=device)
        self.train_label = th.from_numpy(np.load(Y)).to(device=device, dtype=th.int64)
        # self.train_pts = X
        # self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, path_to_data):
        x = th.from_numpy(np.load(path_to_data)).to(device=device)
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p)
        labels = th.argmin(dist, dim=1)
        return self.train_label[labels]


class KNN(NN):

    def __init__(self, X = None, Y = None, k = 3, p = 2):
        self.k = k
        super().__init__(X, Y, p)
    
    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, path_to_data):
        x = th.from_numpy(np.load(path_to_data)).to(device=device)

        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        # candidates = []
        size = 10000
        # top_n = th.zeros((x.shape[0], self.train_pts.shape[0]), dtype=th.float16)
        # top_n_dct = {'indices':[], 'values':[]}
        # candidates = [top_n_dct] * x.shape[0]
        index_topn = []
        value_topn = []
        
        for i in tqdm(range(len(self.train_pts)//size)):
        # dist = distance_matrix(x, self.train_pts, self.p)
            try:
                dist = distance_matrix(x, self.train_pts[i*size:(i+1)*size], self.p)
            except IndexError:
                dist = distance_matrix(x, self.train_pts[i*size:], self.p)
            
            
            knn = dist.topk(self.k, largest=True)#.values
            indices = knn.indices + i*size
            
            index_topn.append(indices)
            value_topn.append(knn.values)
#             if i == 5:
#                 break

        return index_topn, value_topn

knn = KNN(dino_embeds_train_path, dino_labels_train_path, k=10, p=2)

print('kNN counting...')
out1, out2 = knn(dino_embeds_val_path)

good_indices = th.cat(out1, axis=1)
good_values = th.cat(out2, axis=1)


path = '/mnt/lwll/lwll-coral/hrant/cs_patches_256/predictions_knn/'

np.save(os.path.join(path, f'{model}_test38k_10x186_NN.npy'), {
    "good_indices": good_indices,
    "good_values": good_values
})
