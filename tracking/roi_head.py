import torch
from mmcv.runner import BaseModule
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import build_roi_extractor



class RoIHead(BaseModule):

    def __init__(self, featmap_strides=[16], out_channels=768, device='cuda'):
        super().__init__()
        roi_cfg = dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=out_channels,
            featmap_strides=featmap_strides)

        self.track_roi_extractor = build_roi_extractor(roi_cfg).to(device)
        

    def track_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.track_roi_extractor(
            x[:self.track_roi_extractor.num_inputs], rois)
        track_feats = track_feats.reshape((track_feats.shape[0], -1))
        return track_feats

    def extract_bbox_feats(self, x, det_bboxes, img_metas):

        if det_bboxes.size(0) == 0:
            return None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]['scale_factor']).to(det_bboxes.device)
        track_feats = self.track_forward(x, [track_bboxes])

        return track_feats
    
class SimpleRoIHead(torch.nn.Module):

    def __init__(self, featmap_strides=[16], out_channels=768, device='cuda'):
        super().__init__()
        roi_cfg = dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7),
            out_channels=out_channels,
            featmap_strides=featmap_strides)

        self.track_roi_extractor = build_roi_extractor(roi_cfg).to(device)
        

    def forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.track_roi_extractor([x], rois)
        #track_feats = track_feats.reshape((track_feats.shape[0], -1))
        return track_feats