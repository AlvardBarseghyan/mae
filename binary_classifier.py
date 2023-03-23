import argparse

import torch
import torch.nn as nn
import numpy as np

from dataset import BinaryDataset

import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import AUROC

auc = AUROC(task="binary")
class_dct = {   7: ('road', 6111042),
                8: ('sidewalk', 1016635),
                11: ('building', 3822901),
                12: ('wall', 109501),
                13: ('fence', 146414),
                17: ('pole', 203783),
                19: ('traffic light', 34697),
                20: ('traffic sign', 91885),
                21: ('vegetation', 2665158),
                22: ('terrain', 193395),
                23: ('sky', 680522),
                24: ('person', 202630),
                25: ('rider', 22368),
                26: ('car', 1165026),
                27: ('truck', 44584),
                28: ('bus', 38923),
                31: ('train', 38767),
                32: ('motorcycle', 16403),
                33: ('bicycle', 69008)}

class Model(nn.Module):
    def __init__(self, emb_size, num_classes=1):
        super().__init__()
        self.linear = nn.Linear(emb_size, num_classes+1)
        self.activation = nn.Softmax(dim=-1) # sigmoid for binary classificationn


    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)

        return x



class Classifier(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4, experiment=''):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.linear.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        emb, labels = batch
        print()
        print('training step')
        print(emb.shape, labels.shape)
        print(torch.unique(labels, return_counts=True))
        print()
        preds = self.model.forward(emb)

        loss = self.criterion(preds, labels)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        emb, labels = batch  # 
        preds = self.model.forward(emb) # b, 2
        predictions_binary = preds[:, 1]
        loss = self.criterion(preds, labels)

        return [predictions_binary, labels, loss]

    def validation_epoch_end(self, outputs):
        predictions_binary = torch.cat([output[0] for output in outputs])
        labels = torch.cat([output[1] for output in outputs])
        losses = torch.tensor([output[2] for output in outputs])
        # print(predictions_binary.shape)
        # print(labels.shape)
        # print(losses.shape)

        roc = auc(predictions_binary, labels)
        loss = losses.mean()

        self.log('val_loss', loss)
        self.log('val_auc', roc)

def main():

    parser = argparse.ArgumentParser(description='MAE -> Binary classification')

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
    
    parser.add_argument('--embedings_train')
    parser.add_argument('--labels_train')

    parser.add_argument('--embedings_val')
    parser.add_argument('--labels_val')
    parser.add_argument('--training_number', type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--experiment')
    parser.add_argument('--model', default='mae', type=str)

    args = parser.parse_args()

    embeds_train = torch.tensor(np.load(args.embedings_train, allow_pickle=True))
    labels_train = torch.tensor(np.load(args.labels_train, allow_pickle=True))

    embeds_val = torch.tensor(np.load(args.embedings_val, allow_pickle=True))
    labels_val = torch.tensor(np.load(args.labels_val, allow_pickle=True))
    # classes = [7,  8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    classes = [28, 31, 32, 33]
    for cls in classes:
        train_dataset = BinaryDataset(npy_embedings=embeds_train, npy_labels=labels_train,\
            pred_class=cls,training_number=args.training_number, dataset_type='train', \
                shuffle_every_epoch=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,\
            shuffle=True, num_workers=1)

        val_dataset = BinaryDataset(npy_embedings=embeds_val, npy_labels=labels_val,\
            pred_class=cls, training_number=args.training_number)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,\
            shuffle=False, num_workers=4)

        if args.model == 'dino':
            emb_size = 768
        elif args.model == 'mae' or args.model == 'sup_vit':
            emb_size = 1024
        model = Model(emb_size=emb_size, num_classes=1)
        model = Classifier(model=model,learning_rate=args.learning_rate, experiment=f'{class_dct[cls][0]}_{args.experiment}')
        trainer = pl.Trainer(accumulate_grad_batches=args.batch_size, logger=model.aim_logger, \
            enable_checkpointing=True, limit_predict_batches=32, \
            max_epochs=args.epochs, log_every_n_steps=1, accelerator=args.device, callbacks=[model.checkpoint_callback], \
            val_check_interval=int(round(len(train_dataset)/32)))

        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    main()
        
