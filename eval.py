import os
import torch
import argparse
from tqdm import tqdm
from torchmetrics.functional import pairwise_cosine_similarity as cos_dist

from dataset import MAEDataset
from pl_train import LightningMAE

import models_mae

def main():

    parser = argparse.ArgumentParser(description='MAE -> Segmentation task: patch level evaluation')
    parser.add_argument(
        "--dataset_name",
        type=str,
        help='Name of dataset one want to train',
    )
    parser.add_argument(
        '--batch_size',
        default=20,
        type=int,
    )
    parser.add_argument(
        '--device',
        default='cuda',
    )
    parser.add_argument(
        '--checkpoint',
        default='',
        help='absolute path to checkpoint to be loaded',
    )
    parser.add_argument(
        '--intersection_threshold',
        default=0.3,
        type=float,
        help='threshold for patch class',
    )
    parser.add_argument(
        '--server',
        type=str,
        default='c9',
        help='available server names: c9, go',
    )

    args = parser.parse_args()


    if args.dataset_name.lower() == 'fair1m':
        root = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/train1000/'
        path_ann = os.path.join(root, 'few_shot_8.json')
        path_imgs = os.path.join(root, 'images')
        dataset = MAEDataset(path_ann, path_imgs, intersection_threshold=args.intersection_threshold, resize_image=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        num_classes = 5

        root_val = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/val1000/'
        path_ann = os.path.join(root_val, 'few_shot_8.json')
        path_imgs = os.path.join(root_val, 'images')

        dataset_val = MAEDataset(path_ann, path_imgs, resize_image=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=5, shuffle=False)

    elif args.dataset_name.lower() == 'vis_drone' or args.dataset_name.lower() == 'vs':
        root = '/lwll/development/vis_drone/vis_drone_full/train/'
        path_ann = os.path.join('./annotations/', 'few_shot_visdrone.json')
        path_imgs = os.path.join(root, '')

        dataset = MAEDataset(path_ann, path_imgs, intersection_threshold=args.intersection_threshold, resize_image=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        num_classes = 11

    elif args.dataset_name.lower() == 'nightowls':
        path_ann = './annotations/few_shot_8_nightowls.json'
        path_imgs = '/home/ani/nightowls_stage_3/'
        dataset = MAEDataset(path_ann, path_imgs, intersection_threshold=args.intersection_threshold, resize_image=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
        num_classes = 3


    elif args.dataset_name.lower() == 'city_scapes' or args.dataset_name.lower() == 'cs':
        pass # ToDo

    
    if args.server == 'c9':
        chkpt_dir = '/mnt/2tb/hrant/checkpoints/mae_models/mae_visualize_vit_large_ganloss.pth'
    elif args.server == 'go':
        chkpt_dir = './mae_visualize_vit_large_ganloss.pth'
    
    assert args.server in ('c9', 'go'), 'Available server names are c9 and go'

    arch='mae_vit_large_patch16'
    model_mae = getattr(models_mae, arch)()
    if not args.random_init:
        checkpoint = torch.load(chkpt_dir, map_location=args.device)
        msg = model_mae.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

    if args.checkpoint:
        chkpt_dir = args.checkpoint
        model_mae = LightningMAE.load_from_checkpoint(chkpt_dir, model=model_mae)
        model_mae = model_mae.model_mae
    
    model = LightningMAE(model_mae, lr=LEARNING_RATE, l1=L1, num_classes=num_classes)
    if args.device == 'cpu':
        trainer = pl.Trainer(accumulate_grad_batches=4, logger=True, enable_checkpointing=True, limit_predict_batches=args.bathc_size, \
            max_epochs=args.epochs, log_every_n_steps=1, accelerator=args.device)
    else:
        trainer = pl.Trainer(accumulate_grad_batches=4, logger=True, enable_checkpointing=True, limit_predict_batches=args.bathc_size, \
            max_epochs=args.epochs, log_every_n_steps=1, accelerator=args.device, devices=1)

    trainer.fit(model=model, train_dataloaders=dataloader)

if __name__ == '__main__':
    main()