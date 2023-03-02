import argparse

import torch
import torch.nn as nn

from dataset import BinaryDataset

import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks import ModelCheckpoint


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
    def __init__(self, model, experiment=''):
        super().__init__()

        self.model = model
        
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
        optimizer = torch.optim.Adam(self.model.linear.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        emb, labels = batch
        preds = self.model.forward(emb)

        loss = self.criterion(preds, labels)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        emb, labels = batch
        preds = self.model.forward(emb)

        loss = self.criterion(preds, labels)
        self.log('val_loss', loss)

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
    parser.add_argument('--pred_class', type=int)
    parser.add_argument('--training_number', type=int)
    parser.add_argument('--experiment')

    args = parser.parse_args()

    train_dataset = BinaryDataset(npy_embedings=args.embedings_train, npy_labels=args.labels_train,\
         pred_class=args.pred_class,training_number=args.training_number, dataset_type='train', \
            shuffle_every_epoch=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,\
         shuffle=True, num_workers=1)

    # val_dataset = BinaryDataset(npy_embedings=args.embedings_val, npy_labels=args.labels_val,\
    #      pred_class=args.pred_class, training_number=args.training_number)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,\
    #      shuffle=False, num_workers=4)

    model = Model(emb_size=1024, num_classes=1)
    model = Classifier(model=model, experiment=args.experiment)
    trainer = pl.Trainer(accumulate_grad_batches=args.batch_size, logger=model.aim_logger, \
        enable_checkpointing=True, limit_predict_batches=32, \
        max_epochs=args.epochs, log_every_n_steps=1, accelerator=args.device, callbacks=[model.checkpoint_callback])
        # , \
        #     val_check_interval=int(round(len(train_dataset)/32)))

    trainer.fit(model=model, train_dataloaders=train_dataloader)#, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    main()
        
