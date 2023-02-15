import os
import torch
import argparse
from tqdm import tqdm
import pytorch_lightning as pl
from torchmetrics.functional import pairwise_cosine_similarity as cos_dist

from dataset import MAEDataset
from pl_train import LightningMAE

import models_mae


def count_reference(dataloader, model, device, num_classes=5):
    # model.eval()
    reference_sum = torch.zeros((num_classes+1, 1024)).to(device)
    counts = torch.zeros(num_classes+1).to(device)
    for i, ds in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = torch.einsum('nhwc->nchw', ds['image']).to(device)
        img_enc, _, _ = model.forward_encoder(img.float(), mask_ratio=0)
        img_enc = img_enc[:, 1:, :].reshape(-1, img_enc.shape[-1])
        index_labels = ds['indices_labels'].reshape(-1).to(device)
        for j in torch.unique(torch.tensor(index_labels, dtype=int).clone().detach()):
            indices = (index_labels == j).nonzero()
            counts[j] += indices.shape[0]
            new = img_enc[indices].reshape(-1, img_enc.shape[-1])
            # print('class:', j, 'sum:', new.shape, 'shape:', new.sum(dim=0).shape)
            reference_sum[j] += new.sum(dim=0).clone().detach()
            
    return reference_sum, counts

def return_img_embed(dataloader, model):
    # model.eval()
    output = []
    images = []
    labels = []
    for i, ds in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = torch.einsum('nhwc->nchw', ds['image'])
        img_enc, _, _ = model.forward_encoder(img.float(), mask_ratio=0)
        img_enc = img_enc[:, 1:, :] #.reshape(-1, img_enc.shape[-1])
#         img_enc = model(img.float()) #.reshape(-1)
        index_labels = ds['indices_labels'] #.reshape(-1)
        labels.append(index_labels)
        images.append(img.detach())
        output.append(img_enc.detach())

            
    output = torch.cat(output, dim=0)
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    return (output, images, labels)

def main():
    LEARNING_RATE = 1e-4
    L1 = 1
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
    parser.add_argument(
        '--annotation_train'
    )
    parser.add_argument(
        '--annotation_val'
    )
    args = parser.parse_args()


    if args.dataset_name.lower() == 'fair1m':
        root = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/train1000/'
        path_ann = os.path.join(root, args.annotation_train)
        path_imgs = os.path.join(root, 'images')
        dataset = MAEDataset(path_ann, path_imgs, intersection_threshold=args.intersection_threshold, resize_image=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        num_classes = 5

        root_val = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/val1000/'
        path_ann = os.path.join(root_val, args.annotation_val)
        path_imgs = os.path.join(root_val, 'images')

        dataset_val = MAEDataset(path_ann, path_imgs, resize_image=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    elif args.dataset_name.lower() == 'vis_drone' or args.dataset_name.lower() == 'vs':
        root = '/lwll/development/vis_drone/vis_drone_full/train/'
        path_ann = os.path.join('./annotations/', args.annotation_train)
        path_imgs = os.path.join(root, '')

        dataset = MAEDataset(path_ann, path_imgs, intersection_threshold=args.intersection_threshold, resize_image=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        num_classes = 11

        root_val = '/lwll/development/vis_drone/vis_drone_full/train/'
        path_ann = os.path.join('./annotations/', args.annotation_val)
        path_imgs = os.path.join(root_val, '')

        dataset_val = MAEDataset(path_ann, path_imgs, resize_image=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    elif args.dataset_name.lower() == 'nightowls':
        path_ann = f'./annotations/{args.annotation_train}'
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

    checkpoint = torch.load(chkpt_dir, map_location=args.device)
    msg = model_mae.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    chkpt_dir = args.checkpoint
    model_mae = LightningMAE.load_from_checkpoint(chkpt_dir, model=model_mae)
    model_mae.eval()
    model_mae.to(args.device)
    model_mae = model_mae.model_mae
    
    ref_sum, counts = count_reference(
        dataloader=dataloader, 
        model=model_mae, 
        device=args.device,
        num_classes=num_classes)

    ref_mean = (ref_sum.T / counts).T

    embeds, imgs, labels = return_img_embed(dataloader_val, model_mae)
    
if __name__ == '__main__':
    main()