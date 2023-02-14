import os
from tqdm import tqdm
import json
from PIL import Image
import pandas as pd


def coco_dict():
    coco_format = {}
    coco_format['images'] = []
    coco_format['annotations'] = []
    coco_format['categories'] = [{'id': i, 'name': f'{i}', 'supercategory': f'{i}'} for i in range(1, 12)]
    
    
    return coco_format


def annotations(label_filename):
    annotations = pd.read_csv(label_filename, header=None, sep=' ')
    img_boxes = annotations.iloc[:, 1:].values  # [x1, y1, x2, y2]
    img_labels = annotations.iloc[:, 0].values  # label
    return img_boxes, img_labels


def create_coco(path, img_dir, label_dir, from_txt=False):
    if from_txt:
        with open(os.path.join(path, img_dir)) as f:
            x = f.readlines()
        image_names = [i.strip() for i in x]
        
    else:
        image_names = os.listdir(os.path.join(path, img_dir)) # returns list of img names without absolute path
        
    suffix = image_names[0][-4:]
    
    image_names = [x for x in image_names if suffix in x]
    max_img_size, min_img_size = -1, 100000000
    coco_format = coco_dict()
    
    for img_id, img_name in tqdm(enumerate(image_names), total=len(image_names)):
        
        imgs_path_index = img_name.rfind('/')
        img_name, imgs_path = img_name[imgs_path_index+1:], img_name[:imgs_path_index+1]
        
        img_filename = os.path.join(path, imgs_path, img_name)
        
        label_filename = os.path.join(path, label_dir, img_name.replace(suffix, '.txt'))
        
        img = Image.open(img_filename)

        width, height = img.size
        max_img_size = max(max_img_size, height, width)
        min_img_size = min(min_img_size, height, width)
        tmp_img_dct = {
            'file_name': img_name, 
            'height': height,
            'width': width,
            'id': img_id
        }

        coco_format['images'].append(tmp_img_dct)

        img_boxes, img_labels = annotations(label_filename)

        bbox_id = 0
        for boxes, label in zip(img_boxes, img_labels):
            bbox_width, bbox_height = boxes[2] - boxes[0], boxes[3] - boxes[1]
#             print(bbox_width, bbox_height)
            tmp_annotation_dct = {
                'image_id': img_id,
                'category_id': int(label), 
                'bbox': [int(boxes[0]), int(boxes[1]), int(bbox_width), int(bbox_height)],
                'id': bbox_id,
                'iscrowd': 0,
                'area': int(bbox_width * bbox_height)
            }
            coco_format['annotations'].append(tmp_annotation_dct)
            bbox_id += 1

    return coco_format, max_img_size, min_img_size


if __name__ == "__main__":
    parent_path = '/mnt/lwll/lwll-coral/hrant/session_data/tP7Qd3P42Oz9O9K92nDC/base/'
    anns, max_, min_ = create_coco(parent_path, 'train_3.txt', 'labels', True)
    with open('./annotations/few_shot_visdrone.json', 'w') as f:
        json.dump(anns, f)