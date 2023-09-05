import torch
import numpy as np
from torchvision.io import read_image
import cv2
import os

from modeling import Encoder
from tqdm import tqdm

def get_embedings(model_name, annotation_file_name:str, layer=24, folder=''):
    n_patches=14
    if model_name == 'dinov2' or model_name == 'ijepa':
        n_patches = 16
    save_directory = f'./embeddings/cs_patches_256/layers{folder}/'
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
        idx = img['file_name'].find('val') if img['file_name'].find('val') != -1 else img['file_name'].find('train')
        
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
        # image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_CUBIC)

        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(dim=0)
        image = torch.einsum('nhwc->nchw', image)
        black_image = img['black_image']
        #for dino v1 do resize (128, 64)
        patch_dino = cv2.resize(black_image, (n_patches, n_patches), interpolation=cv2.INTER_NEAREST_EXACT).flatten()
        # patch_dino = cv2.resize(black_image, (146, 73), interpolation=cv2.INTER_NEAREST_EXACT).flatten()
        # patch_dino = cv2.resize(black_image, (128, 64), interpolation=cv2.INTER_NEAREST_EXACT).flatten()
        with torch.no_grad():
            img_enc = encoder.forward(image.to('cuda'))
        img_enc = img_enc.cpu().detach()
        img_enc = img_enc.squeeze(dim=0)
        embeddings.append(img_enc)
        labels.append(patch_dino)
    # saving_path_embeds = f'{save_directory}{model_name}_{from_which_annotation}_{layer}_embeds.npy'
    # saving_path_embeds = f'{save_directory}{model_name}_{from_which_annotation}_embeds.npy'

    # saving_path_embeds = f'{save_directory}{model_name}_val_{layer}_embeds.npy'
    # saving_path_labels = f'{save_directory}{model_name}_val_labels.npy'

    # saving_path_embeds = f'./embeddings/cs_full/{model_name}_train_embeds.npy'
    # saving_path_labels = f'./embeddings/cs_full/{model_name}_train_labels.npy'
    
    saving_path_embeds = f'./embeddings/cs_patches_256/{model_name}_embeds_val.npy'
    saving_path_labels = f'./embeddings/cs_patches_256/{model_name}_labels_val.npy'

    # saving_path_embeds = f'./embeddings/ade20k_patches_224/{model_name}_train_embeds.npy'
    # saving_path_labels = f'./embeddings/ade20k_patches_224/{model_name}_train_labels.npy'

    # saving_path_labels = f'{save_directory}{model_name}_{from_which_annotation}_labels.npy'
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    labels = np.concatenate(labels, axis=0)
    print(embeddings.shape, labels.shape)

    print("saving embeds path:", saving_path_embeds)
    # print("saving labels path:", saving_path_labels)
    np.save(saving_path_embeds, embeddings)
    np.save(saving_path_labels, labels)

if __name__ == '__main__':
    # get_embedings('ijepa', './annotations/cs/cs_train.npy')
    # get_embedings('dino', './annotations/ade20k/ade20k_224_train_4.npy')
    # for model in ['dino', 'dinov2', 'mae', 'sup_vit', 'simmim', 'ibot']:
    #     get_embedings(model, './annotations/ade20k/ade20k_224_train_4.npy')
    # get_embedings('dinov2', './annotations/cs/cs4pc_256_train_2.npy')
    # get_embedings('dino', './annotations/cs/cs4pc_256_train_8.npy')
    get_embedings('ibot', './annotations/cs/cs4pc_256_val.npy')
    # get_embedings('dino', './annotations/cs/cs4pc_256_train_4_prime.npy')

    # get_embedings('dino', './annotations/cs/cs4pc_256_val.npy')

    # get_embedings('mae', './annotations/cs/cs4pc_256_train.npy')
    # get_embedings('mae', './annotations/cs/cs4pc_256_val.npy')
    # for model in ('dinov2', 'dino', 'mae' , 'sup_vit'):
    #     for i in [1]: #range(13, 1, -1):
    #         for noise in range(1, 5):
    #             get_embedings(model, f'./annotations/cs/cs4pc_256_val_frequency_noise_{noise}.npy', layer=i, folder=f'_frequency_noise_{noise}')

        # get_embedings('dinov2', './annotations/cs/cs4pc_256_train.npy', layer=i)
    # get_embedings('mae', './annotations/cs/cs4pc_256_train_8.npy')
    # get_embedings('mae', './annotations/cs/cs4pc_256_train_2.npy')
    # get_embedings('mae', './annotations/cs/cs4pc_256_train_4_prime.npy')

    # get_embedings('sup_vit', './annotations/cs/cs4pc_256_train_2.npy')
    # get_embedings('sup_vit', './annotations/cs/cs4pc_256_train_8.npy')
    # get_embedings('sup_vit', './annotations/cs/cs4pc_256_train_4_prime.npy')


    # for i in range(12, 0, -1):
    #     for noise in range(1, 5):
    #         get_embedings('simmim', f'./annotations/cs/cs4pc_256_val_frequency_noise_{noise}.npy', layer=i, folder=f'_frequency_noise_{noise}')

    # for i in range(12, 0, -1):
    #     for blur in range(1, 5):

    # for model in ('dinov2', 'dino', 'mae' , 'sup_vit', 'simmim'):
    #     for i in range(12, 0, -1):
    #         for noise in range(10, 41, 10):
    #             get_embedings(model, f'./annotations/cs/cs4pc_256_val_noise_{noise}.npy', layer=i, folder=f'_noise_{noise}')

    
    # get_embedings('sup_vit', f'./annotations/cs/cs4pc_256_val.npy', layer=12, folder=f'')
    # get_embedings('sup_vit', f'./annotations/cs/cs4pc_256_train.npy', layer=12, folder=f'')
    # get_embedings('sup_vit', f'./annotations/cs/cs4pc_256_train_4_prime.npy', layer=12, folder=f'')
    # get_embedings('sup_vit', f'./annotations/cs/cs4pc_256_train_8.npy', layer=12, folder=f'')
    # get_embedings('sup_vit', f'./annotations/cs/cs4pc_256_train_2.npy', layer=12, folder=f'')



    

