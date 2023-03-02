import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from util.get_dataloader import get_dataloader
from torchmetrics.functional import pairwise_cosine_similarity as cos_dist

from dataset import MAEDataset
from pl_train import LightningMAE, Encoder

import models_mae


def count_reference(dataloader, model, device, num_classes=5):
    # model.eval()
    print(num_classes)
    reference_sum = torch.zeros((num_classes+1, 1024)).to(device)
    counts = torch.zeros(num_classes+1).to(device)
    # print(counts.shape)
    for i, ds in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = torch.einsum('nhwc->nchw', ds['image']).to(device)
        img_enc = model(img.float(), mask_ratio=0) #.reshape(-1, 1024)
        index_labels = ds['indices_labels'].reshape(-1).to(device)
        for j in torch.unique(torch.tensor(index_labels, dtype=int).clone().detach()):
            indices = (index_labels == j).nonzero()
            # print(j, counts.shape)
            counts[j] += indices.shape[0]
            new = img_enc[indices].reshape(-1, img_enc.shape[-1])
            # print('class:', j, 'sum:', new.shape, 'shape:', new.sum(dim=0).shape)
            reference_sum[j] += new.sum(dim=0).clone().detach()
            
    return reference_sum, counts

def return_img_embed(dataloader, model, device):
    # model.eval()
    output = []
    images = []
    labels = []
    filenames = []
    for i, ds in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = torch.einsum('nhwc->nchw', ds['image']).to(device)
        img_enc = model(img.float(), mask_ratio=0)
        # img_enc = model(img.float()) #.reshape(-1)
        img_enc = img_enc.reshape(10, -1, 1024)
        index_labels = ds['indices_labels'] #.reshape(-1)

        labels.append(index_labels)
        images.append(img.detach())
        output.append(img_enc.detach())
        filenames.extend(ds['file_name'])

            
    output = torch.cat(output, dim=0)
    # output = torch.tensor(output)
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    # filenames = torch.cat(filenames, dim=0)
    return (output, images, labels, filenames)

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
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--checkpoint',
        default='/home/hkhachatrian/mae/checkpoints/road_patches_100_new_dataloader/epoch=9.ckpt',
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
        default='go',
        help='available server names: c9, go',
    )
    parser.add_argument('--annotation_train', default='')
    parser.add_argument('--annotation_val', default='')
    parser.add_argument('--save_evaluated_npy')
    parser.add_argument('--loss_type', default='')
    parser.add_argument('--binary_classifier', default=False, type=bool)
    parser.add_argument('--npy_embedings', default='/mnt/lwll/lwll-coral/hrant/cs_patches_256/embeds_val.npy')
    parser.add_argument('--npy_labels', default='/mnt/lwll/lwll-coral/hrant/cs_patches_256/labels_val.npy')
    parser.add_argument('--pred_class', default=7, type=int)

    args = parser.parse_args()

    if args.binary_classifier:
        from binary_classifier import Classifier, Model
        from dataset import BinaryDataset
        from torchmetrics.classification import BinaryF1Score
        model = Model(1024, num_classes=1)
        print(model.__dir__())
        for name in model.named_parameters():
            print(name)
        checkpoint = torch.load(args.checkpoint)
        msg = model.load_state_dict(checkpoint, strict=False)
        # model = Classifier.load_from_checkpoint(args.checkpoint, model=model.model)
        model.to('cuda')
        model.eval()
        embeds = torch.tensor(np.load(args.npy_embedings, allow_pickle=True))
        labels = torch.tensor(np.load(args.npy_labels, allow_pickle=True))

        predictions = []
        gt_labels = []
        print('starting predictions')
        for embed, gt_label in tqdm(zip(embeds, labels), total=len(embeds)):
            preds = model(embed.to('cuda'))
            predictions.append(preds.to('cpu').detach())
            # gt_labels.append(gt_label.to('cpu').detach())

        predictions = torch.cat(predictions, dim=0)
        # gt_labels = torch.cat(gt_labels, dim=0)
        # all_data = torch.cat([predictions.unsqueeze(0), gt_labels.unsqueeze(0)], dim=0)
        torch.save(predictions, './predictions_9_olds.pth')

        metric = BinaryF1Score()
        # print(metric(predictions, gt_labels))
        return




    dataloader, dataloader_val, num_classes, dataset = get_dataloader(dataset_name=args.dataset_name, \
    train_annotation_file=args.annotation_train, val_annotation_file=args.annotation_val, \
    intersection_threshold=args.intersection_threshold, batch_size=args.batch_size, \
    weighted=False, return_dataset=True)
    
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
    model_mae = Encoder(model=model_mae, num_classes=num_classes, classifier=args.loss_type)
    model_mae = LightningMAE.load_from_checkpoint(chkpt_dir, model=model_mae)
    model_mae.eval()
    model_mae.to(args.device)
    model_mae = model_mae.model_mae
    print("class count:", num_classes)
    for name, param in model_mae.named_parameters():
        if 'head' in name:
            print(name)
            
    if args.loss_type == 'contrastive':
        ref_sum, counts = count_reference(
            dataloader=dataloader, 
            model=model_mae, 
            device=args.device,
            num_classes=num_classes)

        ref_mean = (ref_sum.T / counts).T
        ref_mean = ref_mean.nan_to_num(1)
        embeds, imgs, labels, filenames = return_img_embed(dataloader_val, model_mae, args.device)

        ckp = args.checkpoint.split('/')[-1].split('.')[0]
        print(len(filenames), filenames[0])
        output = {}
        output['images'] = []
        print(embeds.shape)
        for i in range(len(embeds)):
            # print(embeds[i].shape, ref_mean.shape)
            tmp_dist = cos_dist(embeds[i], ref_mean)
            tmp_dct = {
                'file_name': filenames[i],
                'patch_labels': tmp_dist.argmax(dim=-1).to('cpu').numpy(),
                'cos_dist': tmp_dist.to('cpu').numpy(),
                'embedings': embeds[i].to('cpu').numpy(),
                'images': imgs[i].to('cpu').numpy(),
                'labels': labels[i].to('cpu').numpy()
            }
            output['images'].append(tmp_dct)


        output['reference_mean'] = ref_mean.to('cpu').numpy()


    else:
        # ds = next(iter(dataloader_val))
        
        # print(img_enc)
        # print(img_enc.shape)
        # print(img_enc.argmax(dim=1))
        output = {}
        output['images'] = []
        for ds in tqdm(dataloader_val, total=len(dataloader_val)):
            img = torch.einsum('nhwc->nchw', ds['image']).to(args.device)
            img_enc = model_mae(img.float(), mask_ratio=0)
            img_enc = img_enc.reshape(args.batch_size, -1, img_enc.shape[-1])
            
            for batch_idx, file_name in enumerate(ds['file_name']):
                tmp_dct = {
                    'file_name': file_name,
                    'patch_labels': img_enc[batch_idx].argmax(dim=-1).to('cpu').numpy(),
                    'images': img[batch_idx].to('cpu').numpy(),
                    'labels': ds['indices_labels'][batch_idx].to('cpu').numpy()
                }
                output['images'].append(tmp_dct)
    print('saving path:', args.save_evaluated_npy)
    np.save(args.save_evaluated_npy, output)

if __name__ == '__main__':
    main()