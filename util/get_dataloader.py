import os
import torch

from dataset import MAEDataset

def get_dataloader(dataset_name, train_annotation_file, val_annotation_file, intersection_threshold, batch_size, weighted=False, return_dataset=False):

    num_classes = 0
    root = ""
    root_val = ""

    if dataset_name.lower() == 'fair1m':
        root = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/train1000/'
        root_val = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/val1000/'

        num_classes = 5

    elif dataset_name.lower() == 'vis_drone' or dataset_name.lower() == 'vd':
        root = '/lwll/development/vis_drone/vis_drone_full/train/'
        root_val = '/lwll/development/vis_drone/vis_drone_full/train/'

        num_classes = 11

    elif dataset_name.lower() == 'nightowls':
        root = '/lwll/development/nightowls_drone/nightowls_full/train/'
        root_val = '/lwll/development/nightowls/nightowls_full/train/'

        num_classes = 3

    elif dataset_name.lower() == 'city_scapes' or dataset_name.lower() == 'cs':
        root = '/mnt/lwll/lwll-coral/hrant/leftImg8bit/train/jena/'
        root_val = '/mnt/lwll/lwll-coral/hrant/leftImg8bit/val/lindau/'

        num_classes = 33  

    path_ann = os.path.join('./annotations/', train_annotation_file)
    path_imgs = os.path.join(root, '')
    dataset = MAEDataset(path_ann, path_imgs, weighted=weighted, intersection_threshold=intersection_threshold, resize_image=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    path_ann = os.path.join('./annotations/', val_annotation_file)
    path_imgs = os.path.join(root_val, '')
    dataset_val = MAEDataset(path_ann, path_imgs, weighted=weighted, resize_image=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    

    if return_dataset:
        return dataloader, dataloader_val, num_classes, dataset
    return dataloader, dataloader_val, num_classes


