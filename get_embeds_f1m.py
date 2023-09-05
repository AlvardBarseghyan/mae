import torch
import numpy as np
from modeling import Encoder
import os
from torchvision.io import read_image
from tqdm import tqdm




def get_embeds(model, annot_file, degradation=None):

    annot_file = np.load(annot_file, allow_pickle=True).item()
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    # shifts = ['_shift1', '_shift2', '_shift3', '_shift4'] # '_no_shift', 
    # shifts = ['_blur_10', '_blur_20', '_blur_30', '_blur_40'] # '_no_shift', 
    if degradation:
        shifts = [degradation]
    else:
        shifts = ['_no_shift', '_shift1', '_shift2', '_shift3', '_shift4'] 
        # shifts = ['_blur_2', '_blur_4', '_blur_6', '_blur_8'] # '_no_shift', 

    encoder = Encoder(model)
    encoder.eval()
    
    for key in shifts:
        embeddings = []
        labels = []
        for img in tqdm(annot_file['images']):

            if key == '_no_shift':
                PATH='./images/f1m/224_8shot/train/'
                saving_path = f'./embeddings/fair1m/{model}_224_8shot'
            else:
                # PATH=f'./images/f1m/224_8shot{key}/train/'
                saving_path = f'./embeddings/fair1m/{model}_224_8shot{key}'

            # path = os.path.join(PATH, img['file_name'])
            path = img['file_name']

            image = read_image(path).permute(1, 2, 0).detach().numpy()
            image = image / 255.
            image = image - imagenet_mean
            image = image / imagenet_std
    
            image = torch.from_numpy(image).float().unsqueeze(0)
            image = torch.einsum('nhwc->nchw', image)

            with torch.no_grad():
                img_enc = encoder.forward(image.to('cuda'))
            
            embeddings.append(img_enc.cpu().detach().squeeze(0).numpy())
            if model == 'dinov2':
                if key != '_no_shift':
                    labels.append(img['patch_labels_16x16'])
                else:
                    labels.append(img[f'patch_labels_16x16_{key[-1]}'])
            else:
                if key != '_no_shift':
                    labels.append(img['patch_labels'])
                else:
                    labels.append(img[f'patch_labels_{key[-1]}'])

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        print(embeddings.shape, labels.shape)
        print(embeddings.shape)

        print("saving embeds path:", saving_path+'_embeddings.npy')
        print("saving labels path:", saving_path+'_labels.npy')

        np.save(saving_path+'_embeddings.npy', embeddings)
        np.save(saving_path+'_labels.npy', labels)


if __name__ == '__main__':
    # get_embeds('ibot', f'./annotations/f1m/224_8shot_shifts.npy', '_no_shift')
    # get_embeds('ibot', f'./annotations/f1m/224_8shot_shifts.npy')
    for model in ['ibot', 'dinov2', 'dino' , 'mae', 'sup_vit', 'simmim']:
        for i in ['0_8', '0_9', '1_1', '1_2']:
            get_embeds(model, f'./annotations/f1m/224_8shot_scale_{i}.npy', f'_scale_{i}')