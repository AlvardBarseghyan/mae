import cv2
import os
import numpy as np
import random
random.seed(6)  # 6ը լավն ա
# np.random.seed(32635)

def get_images_for_collage(train_annotation_file, val_annotaion_file, images_folder, images_per_collage=6):

    train_annotation_images = np.load(train_annotation_file, allow_pickle=True).item()['images']
    val_annotation_images = np.load(val_annotaion_file, allow_pickle=True).item()['images']

    train_samples = random.sample(list(enumerate(train_annotation_images)), images_per_collage)
    val_samples = random.sample(list(enumerate(val_annotation_images)), images_per_collage)

    train_indexes = []
    train_image_names = []
    for train_sample in train_samples:
        train_indexes.append(train_sample[0])
        train_image_names.append(images_folder+'train/images/'+train_sample[1]['file_name'])

    val_indexes = []
    val_image_names = []
    for val_sample in val_samples:
        val_indexes.append(val_sample[0])
        val_image_names.append(images_folder+'val/images/'+val_sample[1]['file_name'])

    return train_indexes, train_image_names, val_indexes, val_image_names

def get_collage(image_names, which_part):
    collage_columns = 2
    collage_rows = 3
    columns = []
    for i in range(collage_rows):
        images_in_row = image_names[i*collage_columns:(i+1)*collage_columns]
        rows = []
        for img in images_in_row:
            rows.append(cv2.resize(cv2.imread(img),(224, 224), interpolation=cv2.INTER_NEAREST_EXACT))
        row_image = np.hstack(rows)
        columns.append(row_image)
    collage = np.vstack(columns)
    cv2.imwrite(f'./static/img/{which_part}.png', collage)


def create_collage(train_annotation_file, val_annotaion_file, images_folder):
    train_indexes, train_image_names, val_indexes, val_image_names = \
        get_images_for_collage(train_annotation_file, val_annotaion_file, images_folder)

    get_collage(train_image_names, 'train')
    get_collage(val_image_names, 'val')
    return train_indexes, val_indexes

# if __name__== '__main__':
#     create_collage('../images/f1m/250_8shot/train/train.npy', '../images/f1m/250_8shot/val/val.npy',\
#                    '../images/f1m/250_8shot/')
    

    