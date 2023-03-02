import os
import cv2
import sys
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision.io import read_image


sys.path.append('../')
from pl_train import Encoder
import models_mae


IMAGE_SIZE=224

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
    device = 'cuda'
    vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    
    return vitb16.to(device)


def forward_dino(self, x):
    x = self.prepare_tokens(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x[:,1:]


def get_image_as_array(path: Path):
    return cv2.imread(str(path)) # returns channel X height x width

def get_image_as_tensor(path: Path):
    return read_image(str(path)) # returns height x width x channel


def get_labeled_image(path: Path):
    return np.array(Image.open(str(path))) # returns h x w


def get_cropped_images(image, labeled_image, height, width):
    # img = cv2.imread(image['file_name'])
    target_size = 256

    y_count = int(np.ceil(height / target_size))
    x_count = int(np.ceil(width / target_size))

    img_pad = np.zeros((y_count * target_size, x_count * target_size, image.shape[2]), np.uint8)
    img_pad[:height, :width] = image

    mask_pad = np.zeros((y_count * target_size, x_count * target_size), np.uint8)
    mask_pad[:height, :width] = labeled_image

    images = []
    labels = []
    for y in range(y_count):
        for x in range(x_count):
            tile = img_pad[y * target_size : (y+1) * target_size, x * target_size : (x+1) * target_size]
            images.append(tile)
            tile_black_image = mask_pad[y * target_size : (y+1) * target_size, x * target_size : (x+1) * target_size]
            tile_patch_labels = cv2.resize(tile_black_image, (14,14), interpolation=cv2.INTER_NEAREST_EXACT).flatten() #.float()

            labels.append(tile_patch_labels)

    return images, labels


def get_image_names(parent_dir):
    print(os.listdir(parent_dir))
    
    folders = [os.path.join(parent_dir, x) for x in os.listdir(parent_dir)]
    res = []
    for folder in folders:
        names = os.listdir(folder)
        for img_name in names:
            res.append(os.path.join(folder, img_name))
    print(res[0])
    return res
    

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


def get_patches(parent_image_dir: str):
    embeddings = []
    labels = []
    encoder = get_model()
    encoder.eval()
    image_names = get_image_names(parent_image_dir)
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
            tmp_emb = encoder(tmp_img.to('cuda'))
            embeddings.append(tmp_emb.to('cpu').detach())
            labels.append(cropped_labels[i])

    embs = torch.cat(embeddings).detach().numpy()
    lbls = np.concatenate(labels, axis=0)

    np.save('/mnt/lwll/lwll-coral/hrant/cs_patches_256/embeds_val.npy', embs)
    np.save('/mnt/lwll/lwll-coral/hrant/cs_patches_256/labels_val.npy', lbls)
    return embs, lbls


def get_dino_resized(img):
    h, w = (img.shape[0] * 14) // 16, (img.shape[1] * 14) // 16
    img = cv2.resize(img, (h, w), interpolation=cv2.INTER_CUBIC)
    return img


def get_dino_patches(parent_image_dir: str):
    embeddings = []
    # labels = []
    encoder = get_dino_model()
    encoder.eval()
    image_names = get_image_names(parent_image_dir)
    for name in tqdm(image_names, total=len(image_names)):
        img = get_image_as_array(name)
        img = torch.tensor(get_dino_resized(img)).permute(2, 0, 1).unsqueeze(0).to(torch.float)
        emb = forward_dino(encoder, img.to('cuda'))
        embeddings.append(emb.to('cpu'))

    embs = torch.cat(embeddings, dim=0).detach().numpy()
    np.save('/mnt/lwll/lwll-coral/hrant/cs_patches_256/dino_embeds_train.npy', embs)


if __name__ == "__main__":
    
    x, y = get_patches('/mnt/lwll/lwll-coral/hrant/leftImg8bit/val/')
    print(x[0], y[0])
    print(x.shape, y.shape)
    # model = get_model()
    # print(np.expand_dims(x[0].transpose(2, 0, 1), axis=0).shape)
    # tmp = np.expand_dims(x[0].transpose(2, 0, 1), axis=0)
    # print(model(tmp).shape)