import os
import cv2
import sys
import h5py
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision.io import read_image


sys.path.append('../')
from pl_train import Encoder
import models_mae


DATASET_NAME = 'fair1m'  #'cs' #
IMAGE_SIZE=224
MODEL= 'sup_vit' # 'dino' 'mae'

def get_model():
    device = 'cuda'
    arch='mae_vit_large_patch16'
    model_mae = getattr(models_mae, arch)()

    chkpt_dir = '../mae_visualize_vit_large_ganloss.pth'
    checkpoint = torch.load(chkpt_dir, map_location=device)
    msg = model_mae.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    encoder = Encoder(model=model_mae, num_classes=33, backbone_freeze=True, classifier='contrastive')
    return encoder.to('cuda')


def get_dino_model():
    print('Loading dino model')
    device = 'cuda'
    vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    
    return vitb16.to(device)

def get_sup_vit_model():
    print('Loading VIT large 16 model')
    device = 'cuda'
    vitb16 = torchvision.models.vit_l_16(weights='IMAGENET1K_V1')
    
    return vitb16.to(device)


def forward_dino(self, x):
    x = self.prepare_tokens(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x[:,1:]

def forward_sup_vit(model, x):
    x = model._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = model.encoder(x)

    return x[:,1:]


def get_image_as_array(path: Path):
    return cv2.imread(str(path)) # returns channel X height x width

def get_image_as_tensor(path: Path):
    return read_image(str(path)) # returns height x width x channel


def get_labeled_image(path: Path):
    return np.array(Image.open(str(path))) # returns h x w


def black_part_proportion(image):
    mask_ind = (image == [0, 0, 0])
    counter = mask_ind.sum()
    tmp = counter / image.size
    # print('black part proportion:', tmp)
    return tmp > -1 # tmp < 0.5

def get_cropped_images(image, labeled_image, height, width):
    # img = cv2.imread(image['file_name'])
    if DATASET_NAME == 'cs':
        target_size = 256
    if DATASET_NAME == 'fair1m':
        target_size = 250

    y_count = int(np.ceil(height / target_size))
    x_count = int(np.ceil(width / target_size))
    
    # print('image_black', (image == [0, 0, 0]).sum()/image.size)

    img_pad = np.zeros((y_count * target_size, x_count * target_size, image.shape[2]), np.uint8)
    img_pad[:height, :width] = image

    mask_pad = np.zeros((y_count * target_size, x_count * target_size), np.uint8)
    mask_pad[:height, :width] = labeled_image

    images = []
    labels = []
    for y in range(y_count):
        for x in range(x_count):
            tile = img_pad[y * target_size : (y+1) * target_size, x * target_size : (x+1) * target_size]
            if DATASET_NAME == 'fair1m':
                if black_part_proportion(tile):
                    images.append(tile)
                    tile_black_image = mask_pad[y * target_size : (y+1) * target_size, x * target_size : (x+1) * target_size]
                    tile_patch_labels = cv2.resize(tile_black_image, (14,14), interpolation=cv2.INTER_NEAREST_EXACT).flatten() #.float()
                    
                    labels.append(tile_patch_labels)

            else:
                images.append(tile)
                tile_black_image = mask_pad[y * target_size : (y+1) * target_size, x * target_size : (x+1) * target_size]
                tile_patch_labels = cv2.resize(tile_black_image, (14,14), interpolation=cv2.INTER_NEAREST_EXACT).flatten() #.float()

                labels.append(tile_patch_labels)

    return images, labels


def get_image_names(parent_dir, from_numpy=False):
    res = []

    if from_numpy:
        image_names = np.load(parent_dir, allow_pickle=True).item()['images']
        for im in image_names:
            res.append(im['file_name'])
        
        return res

    print(os.listdir(parent_dir))
    
    folders = [os.path.join(parent_dir, x) for x in os.listdir(parent_dir)]
    for folder in folders:
        names = os.listdir(folder)
        for img_name in names:
            res.append(os.path.join(folder, img_name))
    print(res[0])
    return res
    

def get_fair1m_label_path(image_path):
    return image_path.replace('images', 'labelTxt').replace('.png', '.json')


def get_label_path(image_path):
    return image_path.replace('leftImg8bit', 'gtFine').replace('.png', '_labelIds.png')


def get_image_normalized(image):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    image = image / 255.
    image = image - imagenet_mean
    image = image / imagenet_std

    return image


def get_patches(parent_image_dir: str, from_numpy=False):
    embeddings = []
    labels = []
    if MODEL == 'dino':
        encoder = get_dino_model()
    if MODEL == 'mae':
        encoder = get_model()
    if MODEL == 'sup_vit':
        encoder = get_sup_vit_model()
    encoder.eval()
    image_names = get_image_names(parent_image_dir, from_numpy)
    for name in tqdm(image_names, total=len(image_names)):
        img = get_image_as_array(name)

        label_name = get_label_path(name)
        lbl_img = get_labeled_image(label_name)
        cropped_imgs, cropped_labels = get_cropped_images(image=img, \
                                        labeled_image=lbl_img, height=img.shape[0], width=img.shape[1])
        
        for i in range(len(cropped_labels)):
            tmp_img = get_image_normalized(cropped_imgs[i])
            tmp_img = np.expand_dims(tmp_img.transpose(2, 0, 1), axis=0)
            tmp_img = torch.tensor(tmp_img, dtype=torch.float).detach()
            if MODEL == 'dino':
                tmp_emb = forward_dino(encoder, tmp_img.to('cuda'))
            elif MODEL == 'mae':
                tmp_emb = encoder(tmp_img.to('cuda'))
            if MODEL == 'sup_vit':
                tmp_emb = forward_sup_vit(encoder, (tmp_img.to('cuda')))

            embeddings.append(tmp_emb.to('cpu').detach())
            labels.append(cropped_labels[i])

    if MODEL == 'dino':
        embs = torch.cat(embeddings, dim=1).squeeze(0).detach().numpy()
    elif MODEL == 'mae':
        embs = torch.cat(embeddings).detach().numpy()
    if MODEL == 'sup_vit':
        embs = torch.cat(embeddings, dim=1).squeeze(0).detach().numpy()
    lbls = np.concatenate(labels, axis=0)

    np.save(f'/mnt/lwll/lwll-coral/hrant/cs_patches_256/{MODEL}_embeds_val_30.npy', embs)
    np.save(f'/mnt/lwll/lwll-coral/hrant/cs_patches_256/{MODEL}_labels_val_30.npy', lbls)
    return embs, lbls


def get_patches_f1m(parent_image_dir: str, annot_file: str):
    print('Dataset loading...')
    image_names = np.load(annot_file, allow_pickle=True)
    # image_names = np.load(annot_file, allow_pickle=True).item()['images']
    for MODEL in ('mae', 'dino', 'sup_vit'):
        labels = []
        print('Model loading...')
        if MODEL == 'dino':
            encoder = get_dino_model()
        if MODEL == 'mae':
            encoder = get_model()
        if MODEL == 'sup_vit':
            encoder = get_sup_vit_model()
        encoder.eval()
        hf = h5py.File(f'/mnt/lwll/lwll-coral/hrant/embeddings/fair1m/f1m_{MODEL}_embeds_train.h5', 'w')
        print('Dataset loading...')
        # image_names = np.load(annot_file, allow_pickle=True).item()['images']
        step = len(image_names)//50
        print(step)

        for j in range(0, len(image_names), step):
            embeddings = []
            for name in tqdm(image_names[j: j+step], total=step):
                img_name = os.path.join(parent_image_dir, name['file_name'])
                img = get_image_as_array(img_name)
                
                lbl_img = name['black_image']
                cropped_imgs, cropped_labels = get_cropped_images(image=img, \
                                                labeled_image=lbl_img, height=img.shape[0], width=img.shape[1])
                for i in range(len(cropped_labels)):
                    tmp_img = get_image_normalized(cropped_imgs[i])
                    tmp_img = np.expand_dims(tmp_img.transpose(2, 0, 1), axis=0)
                    tmp_img = torch.tensor(tmp_img, dtype=torch.float).detach()
                    if MODEL == 'dino':
                        tmp_emb = forward_dino(encoder, tmp_img.to('cuda'))
                    elif MODEL == 'mae':
                        tmp_emb = encoder(tmp_img.to('cuda'))
                    elif MODEL == 'sup_vit':
                        tmp_emb = forward_sup_vit(encoder, (tmp_img.to('cuda')))

                    embeddings.append(tmp_emb.to('cpu').detach())
                    labels.append(cropped_labels[i])            
            if MODEL == 'dino' or 'sup_vit':
                embeddings = torch.cat(embeddings, dim=1).squeeze(0).detach().numpy()
            elif MODEL == 'mae':
                embeddings = torch.cat(embeddings).detach().numpy()
            hf.create_dataset(f'dataset_{j}', data=embeddings)
        hf.close()
        print('cat embeds...')
        # if MODEL == 'dino':
        #     embs = torch.cat(embeddings, dim=1).squeeze(0).detach().numpy()
        # elif MODEL == 'mae':
        #     embs = torch.cat(embeddings).detach().numpy()
        labels = np.concatenate(labels, axis=0)
        print('saving...')
        if 'train' in annot_file:
            # np.save(f'/mnt/lwll/lwll-coral/hrant/embeddings/fair1m/f1m_{MODEL}_embeds_train.npy', embs)
            np.save(f'/mnt/lwll/lwll-coral/hrant/embeddings/fair1m/f1m_{MODEL}_labels_train.npy', labels)
        if 'val' in annot_file:
            # np.save(f'/mnt/lwll/lwll-coral/hrant/embeddings/fair1m/f1m_{MODEL}_embeds_val.npy', embs)
            np.save(f'/mnt/lwll/lwll-coral/hrant/embeddings/fair1m/f1m_{MODEL}_labels_val.npy', labels)
        # np.save(f'/mnt/2tb/alla/embeddings/fair1m/f1m_{MODEL}_embeds_train.npy', embs)
        # np.save(f'/mnt/2tb/alla/embeddings/fair1m/f1m_{MODEL}_labels_train.npy', lbls)
    return embeddings, labels


if __name__ == "__main__":
    
    x, y = get_patches('/home/hkhachatrian/mae/annotations/cs_val.npy', from_numpy=True)
    # print(x[0], y[0])
    # print(x.shape, y.shape)
    # x, y = get_patches_f1m('/mnt/lwll/lwll-coral/FAIR1M/fair1m_1000/train1000/images/', '../annotations/f1m_labeled_5classes_train.npy')
    # print(x, y)
    # print(x.shape, y.shape)
    # x, y = get_patches_f1m('/mnt/lwll/lwll-coral/FAIR1M/fair1m_1000/val1000/images/', '../annotations/f1m_labeled_5classes_val.npy')
    # print(x.shape, y.shape)
    # x, y = get_patches_f1m('/mnt/lwll/lwll-coral/FAIR1M/fair1m_1000/train1000/images/', '../annotations/f1m_labeled_200_train.npy')
    # print(x.shape, y.shape)
    # model = get_model()
    # print(np.expand_dims(x[0].transpose(2, 0, 1), axis=0).shape)
    # tmp = np.expand_dims(x[0].transpose(2, 0, 1), axis=0)
    # print(model(tmp).shape)