import torch
import numpy as np
from torchvision.io import read_image
import cv2
import os
import json
from modeling import Segmenter


from modeling import Encoder
from tqdm import tqdm

def get_embedings(model_name, batch_norm, annotation_file_name:str, layer=24, folder=''):
    n_patches=14
    if model_name == 'dinov2':
        n_patches = 16
    save_directory = f'./embeddings/cs_patches_256/'
    # save_directory = f'./embeddings/cs_patches_256/'
    os.makedirs(save_directory, exist_ok=True)
    from_which_annotation = annotation_file_name.split('/')[-1].split('.')[0]
    # from_which_annotation = from_which_annotation.replace('_cs4pc_256', '')
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    embeddings = []
    labels = []
    encoder = Encoder(model_name, layer=layer)
    encoder.eval()
    annotations = np.load(annotation_file_name, allow_pickle=True).item()
    images = annotations['images']
    for img in tqdm(images):
        # idx = img['file_name'].find('val') if img['file_name'].find('val') != -1 else img['file_name'].find('train')
        
        # image_path = './images/cs/full_ds/' + img['file_name'][idx:]
        # image_path = './images/cs/cs4pc_256/' + img['file_name'][idx:]
        image_path = img['file_name']
        image = read_image(image_path)
        image = image.permute(1, 2, 0).detach().numpy()
        image = image / 255.
        image = image - imagenet_mean
        image = image / imagenet_std
        #comment next line for dino v1
        # image = cv2.resize(image, (2044, 1022), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(dim=0)
        image = torch.einsum('nhwc->nchw', image)
        black_image = img['black_image']
        #for dino v1 do resize (128, 64)
        patch_dino = cv2.resize(black_image, (n_patches, n_patches), interpolation=cv2.INTER_NEAREST_EXACT).flatten()
        # patch_dino = cv2.resize(black_image, (146, 73), interpolation=cv2.INTER_NEAREST_EXACT).flatten()
        with torch.no_grad():
            img_enc = encoder.forward(image.to('cuda'))
        img_enc = img_enc.cpu().detach()
        img_enc = img_enc.squeeze(dim=0)
        img_enc_batch_normed = batch_norm(img_enc)
        embeddings.append(img_enc_batch_normed)
        labels.append(patch_dino)
    saving_path_embeds = f'{save_directory}{model_name}_{from_which_annotation}_batch_normed_embeds.npy'
    # saving_path_embeds = f'{save_directory}{model_name}_{from_which_annotation}_embeds.npy'

    # saving_path_embeds = f'{save_directory}{model_name}_val_{layer}_embeds.npy'
    # saving_path_labels = f'{save_directory}{model_name}_val_labels.npy'
    
    saving_path_labels = f'{save_directory}{model_name}_{from_which_annotation}_batch_normed_labels.npy'
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    labels = np.concatenate(labels, axis=0)
    print(embeddings.shape, labels.shape)

    print("saving embeds path:", saving_path_embeds)
    # print("saving labels path:", saving_path_labels)
    np.save(saving_path_embeds, embeddings)
    np.save(saving_path_labels, labels)

def get_batch_norm(config_path='', checkpoint_path=''):
    with open(config_path) as config:
        hparams= json.load(config)
    backbone_name = hparams['backbone_name']
    num_classes = hparams['num_classes']
    classifier = hparams['decoder_head']
    backbone_freeze = hparams['backbone_freeze']
    read_embeds = hparams['read_embeds']
    
    model = Segmenter(backbone_name=backbone_name, num_classes=num_classes, backbone_freeze=backbone_freeze, classifier_name=classifier, read_embeds=read_embeds)
    
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    batch_norm = model.decoder.linear[0]
    batch_norm.eval()
    
    return batch_norm

if __name__ == '__main__':
    batch_norm = linear_model = get_batch_norm('/mnt/lwll/lwll-coral/hrant/mae_checkpoints/cs4pc_256_mae_layer_12_linear/config.json','/mnt/lwll/lwll-coral/hrant/mae_checkpoints/cs4pc_256_mae_layer_12_linear/val_auc_epoch=244.ckpt')
    get_embedings('mae', batch_norm, './annotations/cs/cs4pc_256_train.npy')
    get_embedings('mae', batch_norm, './annotations/cs/cs4pc_256_val.npy')
