# import copy
# import numpy as np
# #from mmdet.core import bbox2result
# #from mmdet.models import TwoStageDetector
# import mmcv
# import torch
from mmcv import Config
from mmdet.models import build_detector
# from mmdet.apis import init_detector
# from mmcv.runner import get_dist_info, init_dist, load_checkpoint
#from mmdet.apis.inference import show_result
# from ..utils import imshow_tracks, restore_result, track2result
#from mmcv.visualization.image import show_result_pyplot
from mmdet.apis import init_detector, inference_detector
#from mmdet.registry import VISUALIZERS

import mmcv

config_file = 'faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
#model = init_detector(config_file, checkpoint_file, device='cpu')
cfg = Config.fromfile(config_file)
model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
model.cfg = cfg
# model.init_weights('')
# checkpoint = load_checkpoint(model, './faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', map_location='cpu')
#print(model)
img = 'demo.jpeg'
result = inference_detector(model, img)
model.show_result(img, result, out_file='result.jpg')
#model.show_result(img, result, out_file='result.jpg')
# class Detector(TwoStageDetector):
#     def __init__(
#         self,
#         freeze_detector=False,
#         *args,
#         **kwargs
#     ):
#         self.prepare_cfg(kwargs)
#         super().__init__(*args, **kwargs)
#         self.freeze_detector = freeze_detector
#         if self.freeze_detector:
#             self._freeze_detector()

#     def _freeze_detector(self):

#         self.detector = [
#             self.backbone,
#             self.neck,
#             self.rpn_head,
#             self.roi_head.bbox_head,
#         ]

#         for model in self.detector:
#             model.eval()
#             for param in model.parameters():
#                 param.requires_grad = False

#     def show_result(
#         self,
#         img,
#         result,
#         thickness=1,
#         font_scale=0.5,
#         show=False,
#         out_file=None,
#         wait_time=0,
#         backend="cv2",
#         **kwargs
#     ):
#         """Visualize tracking results.

#         Args:
#             img (str | ndarray): Filename of loaded image.
#             result (dict): Tracking result.
#                 The value of key 'track_results' is ndarray with shape (n, 6)
#                 in [id, tl_x, tl_y, br_x, br_y, score] format.
#                 The value of key 'bbox_results' is ndarray with shape (n, 5)
#                 in [tl_x, tl_y, br_x, br_y, score] format.
#             thickness (int, optional): Thickness of lines. Defaults to 1.
#             font_scale (float, optional): Font scales of texts. Defaults
#                 to 0.5.
#             show (bool, optional): Whether show the visualizations on the
#                 fly. Defaults to False.
#             out_file (str | None, optional): Output filename. Defaults to None.
#             backend (str, optional): Backend to draw the bounding boxes,
#                 options are `cv2` and `plt`. Defaults to 'cv2'.

#         Returns:
#             ndarray: Visualized image.
#         """
#         assert isinstance(result, dict)
#         track_result = result.get("track_results", None)
#         bboxes, labels, ids = restore_result(track_result, return_ids=True)
#         img = imshow_tracks(
#             img,
#             bboxes,
#             labels,
#             ids,
#             classes=self.CLASSES,
#             thickness=thickness,
#             font_scale=font_scale,
#             show=show,
#             out_file=out_file,
#             wait_time=wait_time,
#             backend=backend,
#         )
#         return img
