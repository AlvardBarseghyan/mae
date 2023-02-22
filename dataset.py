import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image

IMAGE_SIZE = 224

class MAEDataset(Dataset):
    def __init__(self, coco_json_path, image_path,  patch_size=16, intersection_threshold=0.3, resize_image=False, transforms=False):
        self.image_path = image_path
        self.patch_size = patch_size
        # number of horizontal and vertical columns. in our case it == 14
        self.pixels_in_patch = IMAGE_SIZE // self.patch_size
        self.intersection_threshold = intersection_threshold
        self.resize_image = resize_image
        
        self.anns = np.load(coco_json_path, allow_pickle=True).item()
        # with open(coco_json_path) as f:
        #     self.anns = json.load(f)

        self.transforms = transforms


    def __len__(self):
        return len(self.anns['images'])
    

    def column_number(self, coord):
        return coord // self.patch_size if coord % 224 else (coord-1) // self.patch_size 


    def colnums_to_index(self, x, y):
        return x % self.pixels_in_patch + y * self.pixels_in_patch
    
    
    def bbox_to_index(self, bbox, scale):
        
        ### scaling bounding box coordinates
        box = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        box = self.scale_box(box, scale)
       
        # starting and ending column and row numbers
        x1_col, y1_row, x2_col, y2_row = [self.column_number(bx) for bx in box] 
        
        indices = []
        for i in range(x1_col, x2_col + 1):
            for j in range(y1_row, y2_row + 1):
                indices.append(self.colnums_to_index(i, j))

        return np.array(indices)
    
    
    def np_image_to_base64(self, image, index):
    
        im = cv2.resize(image.permute(1, 2, 0).detach().numpy(), (IMAGE_SIZE, IMAGE_SIZE),\
                        interpolation=cv2.INTER_CUBIC)
        x1, y1, x2, y2 = self.index_to_bbox(index)
        im = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 0), 2)

        return np.array(im)


    def __getitem__(self, idx):
        # print('idx: check', idx)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        img_path, img_id, patch_labels, black_image = [(x['file_name'], x['id'], x['patch_labels'], x['black_image']) for x in self.anns['images']][idx]
        # img_path = '4283__1__0___0.png'
        image = read_image(os.path.join(self.image_path, img_path))
        
        if self.resize_image:
            image = cv2.resize(image.permute(1, 2, 0).detach().numpy(), (IMAGE_SIZE, IMAGE_SIZE),\
                               interpolation=cv2.INTER_CUBIC)
            image = image / 255.
            image = image - imagenet_mean
            image = image / imagenet_std
        else:
            image = image.permute(1, 2, 0).detach().numpy()
        
        if self.transforms:
            # TODO: image cropping as transformation
            pass

        target = {}
        target['image'] = image
        # target['black_image'] = black_image
        target['file_name'] = img_path
        # target['boxes'] = np.array(img_boxes)
        # target['labels'] = np.array(img_labels)
        target['indices_labels'] = np.array(patch_labels)
        # target['image_urls'] = img_urls

        
        return target


if __name__ == "__main__":
    root = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/train1000/'
    path_ann = os.path.join(root, 'few_shot_8.json')
    path_imgs = os.path.join(root, 'images')
    dataset = MAEDataset(path_ann, path_imgs, resize_image=True)
    p = dataset[3]
    print(np.unique(p['file_name'], return_counts=True))
