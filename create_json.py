import os
import cv2
import json
import numpy as np
from tqdm import tqdm

IMAGE_SIZE = 224

class CreateJson:
    def __init__(self, annotation_path, image_path, patch_size=16, intersection_threshold=0.3):
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.patch_size = patch_size
        # number of horizontal and vertical columns. if patch_size==16 => it == 14
        self.patches_in_row = IMAGE_SIZE // self.patch_size
        self.intersection_threshold = intersection_threshold
        
        with open(annotation_path) as f:
            self.annotations = json.load(f)


    def index_to_bbox(self, index):
        x1, x2 = index % self.patches_in_row * self.patch_size, (index % self.patches_in_row + 1) * self.patch_size
        y1, y2 = index // self.patches_in_row * self.patch_size, (index // self.patches_in_row + 1) * self.patch_size

        return np.array([x1, y1, x2, y2])


    def count_colors(self, image):
        colors, counts = np.unique(image.reshape(-1), return_counts=True, axis = 0)
        if len(colors) == 1:
            return int(colors[0])
        
        max_ = (counts[1:] / counts[0]).max()
        if max_ >= self.intersection_threshold:
            return int(colors[counts[1:].argmax() + 1])
        
        return int(colors[counts.argmax()])

    
    def count_colors_cs(self, image):
        colors, counts = np.unique(image.reshape(-1), return_counts=True, axis = 0)
        if len(colors) == 1:
            return int(colors[0])

        return int(colors[counts.argmax()])
    

    def scale_box(self, box, scale):
        x_scale, y_scale = scale        
        scaled = []
        for i in range(0, len(box), 2):
            scaled.append(int(np.round(box[i] * x_scale)))
            scaled.append(int(np.round(box[i+1] * y_scale)))
        
        return np.array(scaled)


    def save_as_json(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f)

    def color_by(self, polygon, iter_by, img_labels, scale, black_image, black_image_resized):
        if polygon == 'seg':
            for seg, label in zip(iter_by, img_labels):
                pts = np.array([[seg[0], seg[1]], [seg[2], seg[3]], [seg[4], seg[5]], [seg[6], seg[7]]])
                black_image = cv2.fillPoly(black_image, [pts], (label, 0))
                seg = self.scale_box(seg, scale)
                pts = np.array([[seg[0], seg[1]], [seg[2], seg[3]], [seg[4], seg[5]], [seg[6], seg[7]]])
                black_image_resized = cv2.fillPoly(black_image_resized, [pts], (label, 0))

            return black_image, black_image_resized

        if polygon == 'box':
            for box, label in zip(iter_by, img_labels):
                pts = np.array([[box[0], box[1]], [box[0] + box[2], box[1] + box[3]]])
                black_image = cv2.rectangle(black_image, pts[0], pts[1], (label, 0), -1)
                box = self.scale_box(box, scale)
                pts = np.array([[box[0], box[1]], [box[0] + box[2], box[1] + box[3]]])
                black_image_resized = cv2.rectangle(black_image_resized, pts[0], pts[1], (label, 0), -1)
            return black_image, black_image_resized

        if polygon == 'polygon':
            for polygon, label in zip(iter_by, img_labels):
                pts = np.array(polygon, np.int32)
                black_image = cv2.fillPoly(black_image, [pts], (label, 0))
                polygon_for_scaling = []
                for p in polygon:
                    polygon_for_scaling.append(p[0])
                    polygon_for_scaling.append(p[1])
                polygon_for_scaling = self.scale_box(polygon_for_scaling, scale)
                polygon = []
                for i in range(0, len(polygon_for_scaling), 2):
                    point = [polygon_for_scaling[i], polygon_for_scaling[i+1]]
                    polygon.append(point)
                pts = np.array(polygon, np.int32)                
                black_image_resized = cv2.fillPoly(black_image_resized, [pts], (label, 0))
            return black_image, black_image_resized


    def fillpoly(self, polygon, path_save, split, dataset_name):
        for anns in tqdm(self.annotations['images']):
            img_path, img_id = anns['file_name'], anns['id']
            img_path = os.path.join(self.image_path, img_path)

            img_labels = [x['category_id'] for x in self.annotations['annotations'] if x['image_id'] == img_id]  # label
            
            if polygon == 'seg':
                img_segmentation = [x['segmentation'][0] for x in self.annotations['annotations'] if x['image_id'] == img_id]
                iter_by = img_segmentation

            elif polygon == 'box':
                img_boxes = [x['bbox'] for x in self.annotations['annotations'] if x['image_id'] == img_id]  # [x1, y1, w, h]
                iter_by = img_boxes

            elif polygon == 'polygon':
                img_polygons = [x['polygon'] for x in self.annotations['annotations'] if x['image_id'] == img_id]  # [x1, y1, w, h]
                iter_by = img_polygons
            
            black_image_resized = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
            black_image = np.zeros((anns['height'], anns['width']))

            x_scale = IMAGE_SIZE / anns['width']
            y_scale = IMAGE_SIZE / anns['height']

            black_image, black_image_resized = self.color_by(polygon, iter_by, img_labels, \
                (x_scale, y_scale), black_image, black_image_resized)

            patch_labels = np.zeros(self.patches_in_row ** 2)
            for i in range(self.patches_in_row ** 2):
                x1, y1, x2, y2 = self.index_to_bbox(i)
                if dataset_name == 'city_scapes':
                    label = self.count_colors_cs(black_image_resized[y1:y2, x1:x2])
                else:
                    label = self.count_colors(black_image_resized[y1:y2, x1:x2])
                patch_labels[i] = label

            anns['file_name'] = img_path
            anns['black_image'] = [list(a) for a in black_image]
            anns['patch_labels'] = list(patch_labels)

        suffix = self.annotation_path.split('/')[-1].replace('.json', '')
        filename = f'{suffix}_{dataset_name}_{split}_thresh_{self.intersection_threshold}_patch_{self.patch_size}.json'
        self.save_as_json(os.path.join(path_save, filename), self.annotations)


if __name__ == "__main__":
    split = 'val'
    dataset_name = 'vis_drone'
    root = '/lwll/development/vis_drone/vis_drone_full/train/'
    root_val = '/lwll/development/vis_drone/vis_drone_full/train/'
    # root = f'/mnt/2tb/hrant/FAIR1M/fair1m_1000/{split}1000'
    # instance = CreateJson(annotation_path=os.path.join(root, 'few_shot_8.json'),\
    #      image_path=os.path.join(root, 'images'),)
    instance = CreateJson(annotation_path=os.path.join('./annotations/' 'few_shot_visdrone_val.json'),\
         image_path=os.path.join(root),)

    instance.fillpoly(path_save='./annotations/', split=f'{split}', dataset_name=dataset_name, polygon='box')