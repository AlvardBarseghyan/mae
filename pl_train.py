import os 
import torch 
import torch.nn as nn
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from loss import ContrastiveLoss
import models_mae
from util.get_dataloader import get_dataloader

from aim.pytorch_lightning import AimLogger
from aim import Image
import numpy as np


# TODO: integrate aim

class Encoder(pl.LightningModule):
    def __init__(self, model, num_classes, backbone_freeze=False, classifier='linear'):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.backbone_freeze = backbone_freeze
        # if backbone_freeze:
        #     print('Frozen backbone')
        #     for param in self.model.parameters():
        #         param.requires_grad = False

        if self.classifier == 'linear' or self.classifier == 'both':
            self.head = nn.Linear(1024, num_classes+1)
            self.activation = nn.Softmax(dim=-1)
        

    def forward(self, x, mask_ratio=0):
        if self.backbone_freeze:
            self.model.eval()
            with torch.no_grad():
                img_enc, mask, _ = self.model.forward_encoder(x, mask_ratio=mask_ratio)
        else:
            img_enc, mask, _ = self.model.forward_encoder(x, mask_ratio=mask_ratio)

        img_enc = img_enc[:, 1:, :].reshape(-1, img_enc.shape[-1])

        if self.classifier == 'contrastive':
            return img_enc

        if self.classifier == 'linear' or self.classifier == 'both':
            img_enc = self.head(img_enc)
            img_enc = self.activation(img_enc)
            return img_enc
        


class LightningMAE(pl.LightningModule):
    def __init__(self, model, weighted=False, loss_type='contrastive', experiment='', l1=0.5, lr=1e-4, num_classes=5, margin=1):
        super().__init__()
        self.model_mae = model
        self.weighted = weighted
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.l1 = l1
        self.lr = lr

        # if backbone_freeze:
        #     print('Frozen backbone')
        #     for param in self.model_mae.parameters():
        #         param.requires_grad = False

        if loss_type == 'contrastive':
            self.criterion = ContrastiveLoss(num_classes=num_classes, margin=margin)
        elif loss_type == 'linear':
            # self.head = nn.Linear(1024, num_classes+1)
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == 'both':
            # self.head = nn.Linear(1024, num_classes+1)
            self.crossentropyloss = nn.CrossEntropyLoss()
            self.contrastiveloss = ContrastiveLoss(num_classes=num_classes, margin=margin)

        self.aim_logger = AimLogger(
            experiment=experiment
        )
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=f'./checkpoints/{experiment}',
            filename='{epoch}',
            verbose=True,
            save_last=True,
            every_n_epochs=5,
            save_top_k=-1
        )
        # self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):
        img = torch.einsum('nhwc->nchw', batch['image'])
        img_enc = self.model_mae(img.float(), mask_ratio=0)

        if self.loss_type == 'linear':
            one_hot_labels = nn.functional.one_hot(batch['indices_labels'].reshape(-1).to(torch.int64), self.num_classes+1)
            total_loss = self.criterion(img_enc, one_hot_labels.to(torch.float32))

        elif self.loss_type == 'contrastive':
            if self.weighted:
                loss = self.criterion(img_enc, batch['indices_labels'].reshape(-1), batch['weighted_labels'].reshape(-1))
            else:
                loss = self.criterion(img_enc, batch['indices_labels'].reshape(-1))

            total_loss = loss[0] + self.l1 * loss[1]
            distance_matrix = loss[2]
            aim_image = Image(distance_matrix)
            
            self.aim_logger.experiment.track(aim_image, "Confusion Matrix")

        elif 'both':
            one_hot_labels = nn.functional.one_hot(batch['indices_labels'].reshape(-1).to(torch.int64), self.num_classes+1)
            cross_loss = self.crossentropyloss(img_enc, one_hot_labels.to(torch.float32))

            if self.weighted:
                loss = self.contrastiveloss(img_enc, batch['indices_labels'].reshape(-1), batch['weighted_labels'].reshape(-1))
            else:
                loss = self.contrastiveloss(img_enc, batch['indices_labels'].reshape(-1))

            total_loss = loss[0] + self.l1 * loss[1] + cross_loss
            
            distance_matrix = loss[2]
            aim_image = Image(distance_matrix)
            
            self.aim_logger.experiment.track(aim_image, "Confusion Matrix")



            # print(f'Iter: {batch_idx}, pos_loss: {loss[0].item()}, neg_loss = {self.l1} * {loss[1].item()}, loss: {total_loss.item()}')
        self.log('train_loss', total_loss)
        print('Epoch', self.current_epoch)
        return total_loss
    
    def validation_step(self, batch, batch_idx):

        img = torch.einsum('nhwc->nchw', batch['image'])
        img_enc = self.model_mae(img.float(), mask_ratio=0)
        if self.loss_type == 'linear':
            one_hot_labels = nn.functional.one_hot(batch['indices_labels'].reshape(-1).to(torch.int64), self.num_classes+1)
            total_loss = self.criterion(img_enc, one_hot_labels.to(torch.float32))

        elif self.loss_type == 'contrastive':
            if self.weighted:
                loss = self.criterion(img_enc, batch['indices_labels'].reshape(-1), batch['weighted_labels'].reshape(-1))
            else:
                loss = self.criterion(img_enc, batch['indices_labels'].reshape(-1))

            total_loss = loss[0] + self.l1 * loss[1]
            distance_matrix = loss[2]
            aim_image = Image(distance_matrix)
            self.aim_logger.experiment.track(aim_image, "Confusion Matrix")

            print(f'Iter: {batch_idx}, pos_loss: {loss[0].item()}, neg_loss = {self.l1} * {loss[1].item()}, loss: {total_loss.item()}')


        elif 'both':
            one_hot_labels = nn.functional.one_hot(batch['indices_labels'].reshape(-1).to(torch.int64), self.num_classes+1)
            cross_loss = self.crossentropyloss(img_enc, one_hot_labels.to(torch.float32))

            if self.weighted:
                print('weigtshdcn ', self.weighted)

                loss = self.contrastiveloss(img_enc, batch['indices_labels'].reshape(-1), batch['weighted_labels'].reshape(-1))
            else:
                loss = self.contrastiveloss(img_enc, batch['indices_labels'].reshape(-1))

            total_loss = loss[0] + self.l1 * loss[1] + cross_loss
            distance_matrix = loss[2]
            aim_image = Image(distance_matrix)
            
            self.aim_logger.experiment.track(aim_image, "Confusion Matrix")

        self.log('val_loss', total_loss)

        return total_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_mae.parameters(), lr=self.lr)
        return optimizer


    def forward(self, img):
        encoded_image, mask, indices = self.model_mae.forward_encoder(img, mask_ratio=0)
        return encoded_image[:, 1:, :]


def main():

    LEARNING_RATE = 1e-4
    L1 = 1
    parser = argparse.ArgumentParser(description='MAE -> Segmentation task')
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
        '--epochs',
        default=100,
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
        '--random_init',
        default=False,
        type=bool,
    )
    parser.add_argument('--annotation_train')
    parser.add_argument('--annotation_val')
    parser.add_argument('--experiment')
    parser.add_argument('--loss_type', default='contrastive')
    parser.add_argument('--weighted', default=False, type=bool)
    parser.add_argument('--backbone_freeze', default=False, type=bool)

    args = parser.parse_args()

    dataloader, dataloader_val, num_classes, dataset = get_dataloader(dataset_name=args.dataset_name, \
    train_annotation_file=args.annotation_train, val_annotation_file=args.annotation_val, \
    intersection_threshold=args.intersection_threshold, batch_size=args.batch_size, \
    weighted=args.weighted, return_dataset=True)
    
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
    
    print(args.weighted, int(args.weighted), args.backbone_freeze, int(args.backbone_freeze))
    encoder = Encoder(model=model_mae, num_classes=num_classes, backbone_freeze=args.backbone_freeze, classifier=args.loss_type)

    model = LightningMAE(model=encoder, weighted=args.weighted, loss_type=args.loss_type, experiment=args.experiment, lr=LEARNING_RATE, l1=L1, num_classes=num_classes)
    if args.device == 'cpu':
        trainer = pl.Trainer(accumulate_grad_batches=32, logger=model.aim_logger, enable_checkpointing=True, limit_predict_batches=args.batch_size, \
            max_epochs=args.epochs, log_every_n_steps=1, accelerator=args.device, val_check_interval=int(round(len(dataset)/args.batch_size)), callbacks=[model.checkpoint_callback])
    else:
        trainer = pl.Trainer(accumulate_grad_batches=32, logger=model.aim_logger, enable_checkpointing=True, limit_predict_batches=args.batch_size, \
            max_epochs=args.epochs, log_every_n_steps=1, accelerator=args.device, devices=1, val_check_interval=int(round(len(dataset)/args.batch_size)), callbacks=[model.checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=dataloader, val_dataloaders=dataloader_val)

if __name__ == '__main__':
    main()