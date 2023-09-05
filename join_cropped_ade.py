import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

n_patches = 14

def main(args):
    imgname_annot = '/mnt/lwll/lwll-coral/hrant/ade20k/ADE20K_2021_17_01/images/ADEChallengeData2016/annotations/validation/'
    reconstructed_imgs = []
    path = os.path.join(args.ann_root, args.cropped_npy)
    n_patches = 14
    if 'dinov2' in path:
        n_patches = 16
    try:
        cropped_images = np.load(path, allow_pickle=True).item()['images']
    except ValueError:
        cropped_images = np.load(path, allow_pickle=True)
    prev_image_name = cropped_images[0]['file_name'].split('/')[-1].split('.')[0]
    save_json = []
    # complete_imgs_path = os.path.join(args.ann_root, args.complete_npy)
    complete_imgs = np.load(args.complete_npy, allow_pickle=True).item()['images']
    current_image = np.zeros([1500, 2500])
    max_i = max_j = 0
    
    for img in tqdm(cropped_images, total=len(cropped_images)):
        image_name = img['file_name'].split('/')[-1].split('.')[0]
        indices = img['file_name'].split('/')[-1].split('.')[1].split('_')[:]
        # print(image_name, img['file_name'])
        split_size, i, j = [int(indices[0]), int(indices[1]), int(indices[2])]
        split_size = n_patches
        if image_name != prev_image_name:
            
            prev_img_info = [x for x in complete_imgs if prev_image_name in x['file_name']][0]
            # w, h = prev_img_info['width'], prev_img_info['height']
            orig_mask = np.array(Image.open(imgname_annot+f'{prev_image_name}.png'))
            h, w = orig_mask.shape[0], orig_mask.shape[1]
            current_image = current_image[:448//split_size, :672//split_size]
            tmp_dct = {
                'file_name': prev_img_info['file_name'],
                'id': prev_img_info['id'], 
                'patch_labels': cv2.resize(current_image, (w, h), interpolation=cv2.INTER_NEAREST_EXACT),
#                 'resized_img': cv2.resize(prev_img_info['patch_labels'].reshape(n_patches, n_patches), (w, h), interpolation=cv2.INTER_NEAREST_EXACT),
                # 'resized_img_from_cropped': cv2.resize(current_image, (w, h), interpolation=cv2.INTER_NEAREST_EXACT),
                'black_image': prev_img_info['black_image'],
                'mask': orig_mask,
#                 'patch_labels_14x14': prev_img_info['patch_labels'].reshape(n_patches, n_patches)
            }
            save_json.append(tmp_dct)
            reconstructed_imgs.append(current_image)

            max_i = max_j = 0
            current_image = np.zeros((1500, 2500))

        # current_image[i * split_size: (i+1) * split_size, j * split_size: (j+1) * split_size] = cv2.resize(img['patch_labels'].reshape(n_patches, n_patches), (split_size, split_size), interpolation=cv2.INTER_NEAREST_EXACT)
        current_image[i * split_size: (i+1) * split_size, j * split_size: (j+1) * split_size] = img['patch_labels'].reshape(n_patches, n_patches)
        # current_image[i * split_size: (i+1) * split_size, j * split_size: (j+1) * split_size] = cv2.resize(img['patch_labels'].reshape(56, 56), (split_size, split_size), interpolation=cv2.INTER_NEAREST_EXACT)
        max_i = max(max_i, i)
        max_j = max(max_j, j)
        prev_image_name = image_name
        
    # current_image = current_image[:h, :w]
    current_image = current_image[:448//split_size, :672//split_size]
    # print(current_image.shape)

    prev_img_info = [x for x in complete_imgs if prev_image_name in x['file_name']][0]
    # w, h = prev_img_info['width'], prev_img_info['height']
    orig_mask = np.array(Image.open(imgname_annot+f'{prev_image_name}.png'))
    h, w = orig_mask.shape[0], orig_mask.shape[1]
    tmp_dct = {
        'file_name': prev_img_info['file_name'],
        'id': prev_img_info['id'], 
        'patch_labels': cv2.resize(current_image, (w, h), interpolation=cv2.INTER_NEAREST_EXACT),
#         'resized_img': cv2.resize(prev_img_info['patch_labels'].reshape(n_patches, n_patches), (w, h), interpolation=cv2.INTER_NEAREST_EXACT),
        # 'resized_img_from_cropped': cv2.resize(current_image, (w, h), interpolation=cv2.INTER_NEAREST_EXACT),
        'black_image': prev_img_info['black_image'],
        'mask': orig_mask,
#         'patch_labels_14x14': prev_img_info['patch_labels'].reshape(n_patches, n_patches)
    }
    save_json.append(tmp_dct)
    reconstructed_imgs.append(current_image)
    
    return save_json, reconstructed_imgs
        

if __name__ == '__main__':
    name = 'instances_train_new_split_city_scapes_train_inter_internearestexact_patch_16.npy'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_root')
    parser.add_argument('--cropped_npy', default='cs4pc_200_train.npy', type=str)
    parser.add_argument('--complete_npy', default=name, type=str)
    parser.add_argument('--save_name_npy', default='cs4pc_upsampled_train.npy')
    parser.add_argument('--save_root')

    args = parser.parse_args()

    json, imgs = main(args)
    print(len(json), json[0]['file_name'])

    np.save(os.path.join(args.save_root, args.save_name_npy), json)