import cv2
import json
import numpy as np
from tqdm import tqdm

path_save='/home/hkhachatrian/mae/images/f1m/fine_grain_500/gt/'
# with open('/mnt/2tb/hrant/FAIR1M/fair1m_1000/val1000/few_shot_200.json') as f:
with open('/mnt/lwll/lwll-coral/FAIR1M/fair1m_1000/val1000/fine_grained500.json') as f:
# with open('/mnt/lwll/lwll-coral/FAIR1M/fair1m_1000/train1000/find_grain8.json') as f:
    data = json.load(f)

for anns in tqdm(data['images'], total=len(data['images'])):
    img_path, img_id = anns['file_name'], anns['id']
    img_labels = [x['category_id'] for x in data['annotations'] if x['image_id'] == img_id]  # label

    img_segmentation = [x['segmentation'][0] for x in data['annotations'] if x['image_id'] == img_id]

    black_image = np.zeros((anns['height'], anns['width']))
    for seg, label in zip(img_segmentation, img_labels):
        pts = np.array([[seg[0], seg[1]], [seg[2], seg[3]], [seg[4], seg[5]], [seg[6], seg[7]]])
        black_image = cv2.fillPoly(black_image, [pts], (label))

    cv2.imwrite(path_save+anns['file_name'], black_image)
#     anns['black_image'] = black_image

# np.save('../annotations/f1m_fine_grain500.npy', data, )
# np.save('../annotations/f1m_fine_grain8.npy', data, )
    # patch_labels = cv2.resize(black_image, (self.patches_in_row, self.patches_in_row), interpolation=cv2.INTER_NEAREST_EXACT).flatten()
