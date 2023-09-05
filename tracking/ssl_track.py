import os
import numpy as np
import torch
from mmdet.core import bbox2result
from mmdet.models import build_detector, build_head
from mmdet.models.detectors.base import BaseDetector
import mmcv
from mmcv.cnn import build_model_from_cfg


from utils import imshow_tracks, restore_result, track2result, init_model
from roi_head import RoIHead
from tracker import Tracker
import sys
sys.path.append('..')
from get_model import get_model_config



class SSLTrack(BaseDetector):

    def __init__(self,
                 cfg,
                 device='cuda',
                 classes=None):
        super().__init__()
        self.CLASSES = classes
        self.cfg = cfg
        self.detector = init_model(cfg, device=device, classes=classes)
        self.feature_model_config = get_model_config(cfg.embed_model)
        patch_size = self.feature_model_config['patch']
        emb_size = self.feature_model_config['emb_size']
        self.track_head = RoIHead(featmap_strides=[patch_size], out_channels=emb_size, device=device)
        self.device = device
        self.embeds_dir = cfg.embeds_dir

    def init_tracker(self):
        self.tracker = Tracker(**self.cfg.tracker)
    # def _freeze_detector(self):

    #     self.detector = [
    #         self.backbone, self.neck, self.rpn_head, self.roi_head.bbox_head
    #     ]
    #     for model in self.detector:
    #         model.eval()
    #         for param in model.parameters():
    #             param.requires_grad = False

    def get_embeds(self, img_metas):
        file_name = img_metas[0]['filename']
        embeds_path = file_name.replace(self.cfg.images_dir, self.cfg.embeds_dir)[:-4] + '.npy'
        embeds = np.load(embeds_path)
        print(embeds.shape)
        r = int(embeds.shape[0] ** 0.5)
        embeds = embeds.reshape((1, r, r, -1))
        embeds = torch.from_numpy(embeds.transpose(0, 3, 1, 2))
        return [embeds.to(self.device)]

    def simple_test(self, img, img_metas, rescale=True):
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0 and hasattr(self, 'tracker') is False:  # for param search
            self.init_tracker()

        x = self.detector.extract_feat(img)
        proposal_list = self.detector.rpn_head.simple_test_rpn(x, img_metas)

       
        det_bboxes, det_labels = self.detector.roi_head.simple_test_bboxes(
            x,
            img_metas,
            proposal_list,
            self.detector.roi_head.test_cfg,
            rescale=rescale)
        #x = tuple(t.detach().cpu() for t in x)
        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]
       
        track_feats = self.track_head.extract_bbox_feats(
            self.get_embeds(img_metas), det_bboxes, img_metas)

        if track_feats is not None:
            bboxes, labels, ids = self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                track_feats=track_feats,
                frame_id=frame_id)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.detector.roi_head.bbox_head.num_classes)


        if track_feats is not None:
            track_result = track2result(
                bboxes, labels, ids,
                self.detector.roi_head.bbox_head.num_classes)
        else:
            track_result = [
                np.zeros((0, 6), dtype=np.float32)
                for i in range(self.detector.roi_head.bbox_head.num_classes)
            ]
        return dict(bbox_results=bbox_result, track_results=track_result)

    def show_result(self,
                    img,
                    result,
                    thickness=1,
                    font_scale=0.5,
                    show=False,
                    out_file=None,
                    wait_time=0,
                    backend='cv2',
                    **kwargs):
        """Visualize tracking results.

        Args:
            img (str | ndarray): Filename of loaded image.
            result (dict): Tracking result.
                The value of key 'track_results' is ndarray with shape (n, 6)
                in [id, tl_x, tl_y, br_x, br_y, score] format.
                The value of key 'bbox_results' is ndarray with shape (n, 5)
                in [tl_x, tl_y, br_x, br_y, score] format.
            thickness (int, optional): Thickness of lines. Defaults to 1.
            font_scale (float, optional): Font scales of texts. Defaults
                to 0.5.
            show (bool, optional): Whether show the visualizations on the
                fly. Defaults to False.
            out_file (str | None, optional): Output filename. Defaults to None.
            backend (str, optional): Backend to draw the bounding boxes,
                options are `cv2` and `plt`. Defaults to 'cv2'.

        Returns:
            ndarray: Visualized image.
        """
        assert isinstance(result, dict)
        track_result = result.get('track_results', None)
        bboxes, labels, ids = restore_result(track_result, return_ids=True)
        img = imshow_tracks(
            img,
            bboxes,
            labels,
            ids,
            classes=self.CLASSES,
            thickness=thickness,
            font_scale=font_scale,
            show=show,
            out_file=out_file,
            wait_time=wait_time,
            backend=backend)
        return img

    def extract_feat(self, imgs):
        return super().extract_feat(imgs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return super().aug_test(imgs, img_metas, **kwargs)
