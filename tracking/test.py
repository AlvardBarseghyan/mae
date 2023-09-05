import argparse
import os
import numpy as np
import mmcv
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.datasets import build_dataset
from collections import defaultdict
from tqdm import tqdm

import datasets
from datasets import build_dataloader
from utils import init_model
from utils import file_utils
from ssl_track import SSLTrack



def main():
    cfg = Config.fromfile('configs/tracker_rcnn_r50_bdd.py')
    # from mmdet.datasets import DATASETS
    # # import pdb
    # # pdb.set_trace()
    # if cfg.get('USE_MMDET', False):
    #     from mmdet.apis import multi_gpu_test, single_gpu_test
    #     from mmdet.models import build_detector as build_model
    #     from mmdet.datasets import build_dataloader

    print(f'Tracking using {cfg.embed_model} features of {cfg.img_size} size images') 
    cfg.data.test.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = SSLTrack(cfg=cfg,
                    device=cfg.device,
                    classes=dataset.CLASSES)
    
    model.CLASSES = dataset.CLASSES
    model.eval()
    #embeds_dir = os.path.join(cfg.input_data_root,  f'embeds_{cfg.embed_model}_{cfg.img_size}{cfg.suff}/track/val/')
    results = defaultdict(list)
    pbar = tqdm(enumerate(data_loader), total=len(dataset), unit="image", desc=f'Inference')
    for i, data in pbar:
        img_metas = data['img_metas'][0].data[0]
        with torch.no_grad():
            result = model.simple_test(img=data['img'][0].float().to(cfg.device), 
                                       img_metas=img_metas,
                                       rescale=True)
        for k, v in result.items():
            results[k].append(v)


    import pickle
    with open(f'results/results_{cfg.embed_model}_{cfg.img_size}{cfg.suff}_new.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('Results saved successfully to file')
    
    # with open(f'results/results_{cfg.embed_model}_{cfg.img_size}_cos.pkl', 'rb') as f:
    #     results = pickle.load(f)


    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='track'))
    print(dataset.evaluate(results, **eval_kwargs))


if __name__ == '__main__':
    main()
