import os
import torch

from dataset import SegDataset, EmbeddingsDataset

def get_dataloader(dataset_name, train_annotation_file, val_annotation_file, batch_size, n_patches=14,  weighted=False, resize_image=True, read_embeds=False):

    if read_embeds:
        data_folder = os.path.basename(train_annotation_file[0]).split('.')[0]
        idx = data_folder.rfind('_')
        data_folder = data_folder[:idx]

    else:
        print()
        data_folder = os.path.basename(train_annotation_file).split('.')[0]
        idx = data_folder.rfind('_')
        data_folder = data_folder[:idx]

    num_classes = 0
    root_train = f'./images/{dataset_name}/{data_folder}/train/images/'
    root_val = f'./images/{dataset_name}/{data_folder}/val/images/'
    num_classes = 33 if dataset_name == 'cs' else 5

    if dataset_name.lower() == 'vis_drone' or dataset_name.lower() == 'vd':
        root_train = '/lwll/development/vis_drone/vis_drone_full/train/'
        root_val = '/lwll/development/vis_drone/vis_drone_full/train/'

        num_classes = 11

    elif dataset_name.lower() == 'nightowls':
        root_train = '/lwll/development/nightowls_drone/nightowls_full/train/'
        root_val = '/lwll/development/nightowls/nightowls_full/train/'

        num_classes = 3

    elif dataset_name.lower() == 'ade20k':
        root_train = 'whatever'
        root_val = 'whatever'

        num_classes = 150

    if read_embeds:
        assert len(train_annotation_file) == 2, "This variable should be a tuple of embedding and label file paths"
        assert len(val_annotation_file) == 2, "This variable should be a tuple of embedding and label file paths"

        train_embeds_path, train_labels_path = train_annotation_file
        train_dataset = EmbeddingsDataset(embeds_path=train_embeds_path, labels_path=train_labels_path)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_embeds_path, val_labels_path = val_annotation_file
        val_dataset = EmbeddingsDataset(embeds_path=val_embeds_path, labels_path=val_labels_path)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    else:
        # path_ann = os.path.join('./annotations/', train_annotation_file)
        path_imgs = os.path.join(root_train, '')
        path_ann = train_annotation_file
        train_dataset = SegDataset(path_ann, path_imgs, weighted=weighted, n_patches=n_patches, resize_image=resize_image)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # path_ann = os.path.join('./annotations/', val_annotation_file)
        path_imgs = os.path.join(root_val, '')
        path_ann = val_annotation_file
        val_dataset = SegDataset(path_ann, path_imgs, n_patches=n_patches, weighted=weighted, resize_image=resize_image)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, num_classes, len(train_dataset)


