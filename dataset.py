import os
import cv2
import json
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
from collections import Counter
import torch

IMAGE_SIZE = 224
np.random.seed(32635)

class BinaryDataset(Dataset):
    def __init__(self, npy_embedings, npy_labels, pred_class, training_number=None, dataset_type='val', shuffle_every_epoch=True):
        self.dataset_type = dataset_type
        self.shuffle_every_epoch = shuffle_every_epoch
        self.pred_class = pred_class

        if dataset_type == 'val':
            print('Loading embeddings')
            self.all_embeds = npy_embedings
            self.all_labels = npy_labels

        else:
            self.training_number = training_number
            self.counter = 0
            print('Loading embeddings')
            print(f'Number of patches: {self.training_number} for class #{self.pred_class} shuffle {self.shuffle_every_epoch}')

            self.all_embeds = torch.tensor(npy_embedings)
            self.all_labels = torch.tensor(npy_labels)
            class_indices = np.where(self.all_labels == pred_class)[0]
            self.other_indices = np.where(self.all_labels != pred_class)[0]
            np.random.shuffle(class_indices)
            self.positive_indices = class_indices[:training_number]
            self.positive_embedings = torch.tensor(self.all_embeds[self.positive_indices])
            np.random.shuffle(self.other_indices)
            indices = np.concatenate((self.positive_indices, self.other_indices[:self.training_number]), axis=0)
            self.embedings = torch.tensor(self.all_embeds[indices])
            self.labels = torch.tensor(self.all_labels[indices])



    def __getitem__(self, idx):
        if self.dataset_type == 'val':
            binary_label = torch.tensor(int(self.all_labels[idx].item()==self.pred_class))

            return   self.all_embeds[idx], binary_label

        else:
            # print(self.counter)
            if self.shuffle_every_epoch and self.counter + 1 == self.__len__():
                print(self.counter, "new init")
                print('Everyday Im shuffling')
                # print(f'Number of patches: {self.training_number} for class #{self.pred_class} shuffle {self.shuffle_every_epoch}')
                np.random.shuffle(self.other_indices)
                indices = np.concatenate((self.positive_indices, self.other_indices[:self.training_number]), axis=0)
                self.embedings = self.all_embeds[indices]
                self.labels = self.all_labels[indices]
                self.counter = 0

            binary_label = torch.tensor(int(self.labels[idx].item()==self.pred_class))
            self.counter += 1
            return  self.embedings[idx], binary_label

    def __len__(self):
        if self.dataset_type == 'val':
            return len(self.all_embeds)

        return 2*len(self.positive_embedings)


class SegDataset(Dataset):
    def __init__(self, annot_path, image_path, weighted=False, n_patches=14, resize_image=False, transforms=False):
        self.image_path = image_path

        # number of horizontal and vertical columns. in our case it == 14
        self.n_patches = n_patches
        self.weighted = weighted
        self.resize_image = resize_image
        
        self.anns = np.load(annot_path, allow_pickle=True).item()

        self.transforms = transforms
        if weighted:
            self.stats_mask, self.stats_patch_labels = self.get_stats()
            self.weights = {key: 1/ val**0.5 if val !=0 else val for key, val in self.stats_mask.items()}
        

    def get_stats(self):
        stats_mask = Counter()
        stats_patch_labels = Counter()
        for image in self.anns['images']:
            mask = np.array(image['black_image'])
            patch_labels = np.array(image['patch_labels'])
            for cl in range(0, 35):
                stats_mask[cl] += np.sum(mask==cl)
                stats_patch_labels[cl] += np.sum(patch_labels==cl)

        return stats_mask, stats_patch_labels

    def __len__(self):
        return len(self.anns['images'])
    

    def column_number(self, coord):
        return coord // self.patch_size if coord % 224 else (coord-1) // self.patch_size 


    def colnums_to_index(self, x, y):
        return x % self.n_patches + y * self.n_patches
    
    
    def f(self, bbox, scale):
        
        ### scaling bounding box coordinates
        box = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        box = self.scale_box(box, scale)
       
        # starting and ending column and row numbers
        x1_col, y1_row, x2_col, y2_row = [self.column_number(bx) for bx in box] 
        
        indices = []
        for i in range(x1_col, x2_col + 1):
            for j in range(y1_row, y2_row + 1):
                indices.append(self.colnums_to_index(i, j))

        return np.array(indices)
    
    
    def np_image_to_base64(self, image, index):
    
        im = cv2.resize(image.permute(1, 2, 0).detach().numpy(), (IMAGE_SIZE, IMAGE_SIZE),\
                        interpolation=cv2.INTER_CUBIC)
        x1, y1, x2, y2 = self.index_to_bbox(index)
        im = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 0), 2)

        return np.array(im)


    def __getitem__(self, idx):
        # print('idx: check', idx)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        img_name, patch_labels, black_image = [(x['file_name'], x['patch_labels'], x['black_image']) for x in self.anns['images']][idx]
        # img_path = '4283__1__0___0.png'
        img_name = os.path.basename(img_name)
        image = read_image(os.path.join(self.image_path, img_name))
        
        if self.resize_image:
            image = cv2.resize(image.permute(1, 2, 0).detach().numpy(), (IMAGE_SIZE, IMAGE_SIZE),\
                               interpolation=cv2.INTER_CUBIC)
            image = image / 255.
            image = image - imagenet_mean
            image = image / imagenet_std
        else:
            image = image.permute(1, 2, 0).detach().numpy()
            image = image / 255.
            image = image - imagenet_mean
            image = image / imagenet_std
        
        target = {}

        # target['file_name'] = os.path.join(self.image_path, img_name)
        target['image'] = image
        target['black_image'] = black_image
        target['indices_labels'] = np.array(patch_labels)

        #TODO for upsampling and different patch_sizes
        if self.n_patches != 14:
            target[f'patch_{self.n_patches}x{self.n_patches}'] = cv2.resize(black_image, (self.n_patches, self.n_patches), interpolation=cv2.INTER_NEAREST_EXACT).flatten()
        # target['patch_dino'] = cv2.resize(black_image, (512, 256), interpolation=cv2.INTER_NEAREST_EXACT).flatten()


        if self.weighted:
            weighted_labels = np.array([self.weights[i] for i in patch_labels])
            target['weighted_labels'] = weighted_labels
        
        # target['image_urls'] = img_urls

        
        return target
    
class EmbeddingsDataset(Dataset):
    def __init__(self, embeds_path, labels_path):
        print(f"Downloading {embeds_path}...")
        self.embeddings = np.load(embeds_path, allow_pickle=True)
        self.labels = np.load(labels_path, allow_pickle=True)

        
    def __len__(self):
        return self.embeddings.shape[0]


    def __getitem__(self, idx):
        return {'image': self.embeddings[idx, :], 'indices_labels': self.labels[idx]}



if __name__ == "__main__":
    root = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/train1000/'
    path_ann = os.path.join(root, 'few_shot_8.json')
    path_imgs = os.path.join(root, 'images')
    dataset = SegDataset(path_ann, path_imgs, resize_image=True)
    p = dataset[3]
    print(np.unique(p['file_name'], return_counts=True))
