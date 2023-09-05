import warnings
import os
# import sys
# sys.path.append('/home/hkhachatrian/mae/tracking')
import mmcv
import numpy as np
import cv2
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.apis import inference_detector




def init_model(config, checkpoint=None, device='cuda:0', classes=None, cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=None)
    if checkpoint is None:
        checkpoint = config['checkpoint_path']
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc, strict=False)
        if classes is not None:
            model.CLASSES = classes
        else:
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


# def inference_model(model, img, frame_id):
#     cfg = model.cfg
#     device = next(model.parameters()).device  # model device
#     # prepare data
#     if isinstance(img, np.ndarray):
#         # directly add img
#         data = dict(img=img, frame_id=frame_id, img_prefix=None)
#         cfg = cfg.copy()
#         # set loading pipeline type
#         cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
#     else:
#         # add information into dict
#         data = dict(
#             img_info=dict(filename=img), frame_id=frame_id, img_prefix=None)
#     # build the data pipeline
#     test_pipeline = Compose(cfg.data.test.pipeline)
#     data = test_pipeline(data)
#     data = collate([data], samples_per_gpu=1)
#     if next(model.parameters()).is_cuda:
#         # scatter to specified GPU
#         data = scatter(data, [device])[0]
#     else:
#         for m in model.modules():
#             assert not isinstance(
#                 m, RoIPool
#             ), 'CPU inference with RoIPool is not supported currently.'
#         # just get the actual data from DataContainer
#         data['img_metas'] = data['img_metas'][0].data
#     # forward the model
#     with torch.no_grad():
#         result = model(return_loss=False, rescale=True, **data)
#     return result

def inference_model(model, img, frame_id):
    cfg = model.cfg
    data = dict(
        img_info=dict(filename=img), frame_id=frame_id, img_prefix=None)
    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)

    img_metas = [data['img_metas'][0].data]
    img=data['img'][0].float().to(cfg.device).unsqueeze(0)
    with torch.no_grad():
        result = model.simple_test(img=img, 
                                img_metas=img_metas,
                                rescale=True)

    return result


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       fig_size=(15, 10),
                       title='result',
                       block=True,
                       wait_time=0):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        title (str): Title of the pyplot figure.
        block (bool): Whether to block GUI. Default: True
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    warnings.warn('"block" will be deprecated in v2.9.0,'
                  'Please use "wait_time"')
    warnings.warn('"fig_size" are deprecated and takes no effect.')
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241))


if __name__ == '__main__':

    config_file = '../configs/faster_rcnn_r50_fpn_1x_coco_bdd.py'
    # checkpoint_file = '../detector/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth'
    # #checkpoint_file = '../detector/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # model = init_model(config_file, checkpoint=checkpoint_file, device='cpu')
    # #model.cfg = cfg
    
    # #print(model)
    # #torch.save(model.state_dict(), 'frcnn_r50_fpn_12e_bdd100k.pth')
    # img = 'detector/b1d4b62c-60aab822.jpg'
    # # img_np = cv2.imread(img)
    # # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    # # inp = torch.from_numpy(img_np).unsqueeze(0).float()
    # # a = model.extract_feat(inp)
    # # b = []
    # # model_parameters = filter(lambda p: p.requires_grad, model.roi_head.bbox_roi_extractor.parameters())
    # # params = sum([np.prod(p.size()) for p in model_parameters])
    # # print(params)
    # result = inference_detector(model, img)
    # model.show_result(img, result, out_file='detector/result.jpg')
    img = '../detector/b1d4b62c-60aab822.jpg'
    # img_np = cv2.imread(img)
    # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    config = mmcv.Config.fromfile(config_file)
    data = dict(
            img_info=dict(filename=img), frame_id=2, img_prefix=None)
    # build the data pipeline
    test_pipeline = Compose(config.data.test.pipeline)
    data = test_pipeline(data)
    import pdb
    pdb.set_trace()
    print(data.keys())