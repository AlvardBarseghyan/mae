import os 
import torch 
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

class LightningMAE(pl.LightningModule):
    def __init__(self, model, experiment='', l1=0.5, lr=1e-4, num_classes=5, margin=1):
        super().__init__()
        self.model_mae = model
        self.num_classes = num_classes
        self.l1 = l1
        self.lr = lr
        self.criterion = ContrastiveLoss(num_classes=num_classes, margin=margin)
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
        print("Getting image embeddings...")
        img_enc, mask, _ = self.model_mae.forward_encoder(img.float(), mask_ratio=0)
        img_enc = img_enc[:, 1:, :].reshape(-1, img_enc.shape[-1])
        print("Loss counting...")
        loss = self.criterion(img_enc, batch['indices_labels'].reshape(-1), batch['weighted_labels'].reshape(-1))
        total_loss = loss[0] + self.l1 * loss[1]
        distance_matrix = loss[2]
        aim_image = Image(distance_matrix)
        self.log('train_loss', total_loss)
        self.aim_logger.experiment.track(aim_image, "Confusion Matrix")
        print(f'Iter: {batch_idx}, pos_loss: {loss[0].item()}, neg_loss = {self.l1} * {loss[1].item()}, loss: {total_loss.item()}')

        return total_loss
    
    def validation_step(self, batch, batch_idx):

        img = torch.einsum('nhwc->nchw', batch['image'])
        print("Getting image embeddings...")
        img_enc, mask, _ = self.model_mae.forward_encoder(img.float(), mask_ratio=0)
        img_enc = img_enc[:, 1:, :].reshape(-1, img_enc.shape[-1])
        print("Val loss counting...")
        loss = self.criterion(img_enc, batch['indices_labels'].reshape(-1), batch['weighted_labels'].reshape(-1))
        total_loss = loss[0] + self.l1 * loss[1]
        distance_matrix = loss[2]
        print(np.array(distance_matrix).shape)
        aim_image = Image(distance_matrix)
        self.aim_logger.experiment.track(aim_image, "Confusion Matrix val")
        self.log('val_loss', total_loss)
        print(f'Iter: {batch_idx}, val_pos_loss: {loss[0].item()}, val_neg_loss = {self.l1} * {loss[1].item()}, val_loss: {total_loss.item()}')

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
    parser.add_argument(
        '--annotation_train'
    )
    parser.add_argument(
        '--annotation_val'
    )
    parser.add_argument(
        '--experiment'
    )

    args = parser.parse_args()

    dataloader, dataloader_val, num_classes, dataset = get_dataloader(args.dataset_name, args.annotation_train, args.annotation_val, args.intersection_threshold, args.batch_size, True)
    
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
    
    model = LightningMAE(model_mae, experiment=args.experiment, lr=LEARNING_RATE, l1=L1, num_classes=num_classes)
    if args.device == 'cpu':
        trainer = pl.Trainer(accumulate_grad_batches=32, logger=model.aim_logger, enable_checkpointing=True, limit_predict_batches=args.batch_size, \
            max_epochs=args.epochs, log_every_n_steps=1, accelerator=args.device, val_check_interval=int(round(len(dataset)/args.batch_size)), callbacks=[model.checkpoint_callback])
    else:
        trainer = pl.Trainer(accumulate_grad_batches=32, logger=model.aim_logger, enable_checkpointing=True, limit_predict_batches=args.batch_size, \
            max_epochs=args.epochs, log_every_n_steps=1, accelerator=args.device, devices=1, val_check_interval=int(round(len(dataset)/args.batch_size)), callbacks=[model.checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=dataloader, val_dataloaders=dataloader_val)

if __name__ == '__main__':
    main()