import os
from argparse import Namespace
import json

from torchmetrics import AUROC
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from util.get_dataloader import get_dataloader
from modeling import Segmenter
from loss import ContrastiveLoss


class SegTrainer(Segmenter):
    def __init__(self, hparams: Namespace) -> None:

        hparams_dict = vars(hparams)

        self.experiment_name = hparams_dict['experiment_name']
        self.lr = hparams_dict['lr']
        self.loss_type = hparams_dict['loss_type']
        self.weighted = hparams_dict['weighted']
        self.l1 = hparams_dict['l1']
        self.read_embeds = hparams_dict['read_embeds']
        if hparams_dict['read_embeds']:
            train_annotation_file = (
                hparams_dict['train_embedds'], hparams_dict['train_embedds_labels'])
            val_annotation_file = (
                hparams_dict['val_embedds'], hparams_dict['val_embedds_labels'])
        else:
            train_annotation_file = hparams_dict['annotation_train']
            val_annotation_file = hparams_dict['annotation_val']

        self.train_dataloader, self.val_dataloader, self.num_classes, dataset_len = get_dataloader(dataset_name=hparams_dict['dataset_name'],
                                                                                                   train_annotation_file=train_annotation_file, val_annotation_file=val_annotation_file,
                                                                                                   n_patches=hparams_dict['n_patches'], batch_size=hparams_dict['batch_size'],
                                                                                                   weighted=hparams_dict['weighted'], resize_image=hparams_dict['resize_image'], 
                                                                                                   read_embeds=hparams_dict['read_embeds'])

        super().__init__(hparams_dict['backbone_name'], self.num_classes,
                         hparams_dict['emb_size'], hparams_dict['decoder_head'], 
                         hparams_dict['backbone_freeze'], hparams_dict['read_embeds'])
        # , hparams_dict['bilinear_upsampling'], hparams_dict['deconvolution'],

        self.auc = AUROC(task="multiclass", num_classes=self.num_classes+1)
        self.soft_activation = torch.nn.Softmax(dim=-1)

        if self.loss_type == 'contrastive':
            self.criterion = ContrastiveLoss(
                num_classes=self.num_classes, margin=hparams_dict['margin'])
        elif self.loss_type == 'ce':
            # self.head = troch.nn.Linear(1024, num_classes+1)
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.loss_type == 'both':
            # self.head = nn.Linear(1024, num_classes+1)
            self.crossentropyloss = torch.nn.CrossEntropyLoss()
            self.contrastiveloss = ContrastiveLoss(
                num_classes=self.num_classes, margin=hparams_dict['margin'])
        elif self.loss_type == 'upsample':
            self.criterion = torch.nn.CrossEntropyLoss()

        self.checkpoint_callback_loss = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'./checkpoints/{self.experiment_name}',
            filename='val_loss_{epoch}',
            verbose=True,
            save_last=True,
            every_n_epochs=5,
            save_top_k=-1,
            mode='min'
        )
        self.checkpoint_callback_acc = ModelCheckpoint(
            monitor='val_auc',
            dirpath=f'./checkpoints/{self.experiment_name}',
            filename='val_auc_{epoch}',
            verbose=True,
            save_last=True,
            every_n_epochs=5,
            save_top_k=-1,
            mode='max'
        )

        # self.save_hyperparameters(ignore=['model'])
        self.experiment = os.path.join('./checkpoints', self.experiment_name)
        os.makedirs(self.experiment, exist_ok=True)
        wandb_logger = WandbLogger(
            project='Few-shot segmentation', name=self.experiment_name, config=hparams_dict)
        wandb_logger.experiment.config["num_classes"] = self.num_classes
        hparams_dict['num_classes'] = self.num_classes
        config_save_path = self.experiment + '/config.json'
        with open(config_save_path, 'w') as cf:
            json.dump(hparams_dict, cf)

        self.trainer = pl.Trainer(accumulate_grad_batches=32, logger=wandb_logger, enable_checkpointing=True,
                                  limit_predict_batches=hparams_dict[
                                      'batch_size'], max_epochs=hparams_dict['epochs'], log_every_n_steps=1,
                                  accelerator=hparams_dict['device'], val_check_interval=int(
                                      round(dataset_len/hparams_dict['batch_size'])),
                                  callbacks=[self.checkpoint_callback_loss, self.checkpoint_callback_acc], )
        os.environ['WANDB_DISABLED'] = 'true'

    def training_step(self, batch, batch_idx):
        if self.read_embeds:
            img = batch['image']
        else:
            # maybe move it to dataset?
            img = torch.einsum('nhwc->nchw', batch['image'])
        img_enc = self.forward(img.float())
        # print('######################')
        # print("loss type", self.loss_type)

        if self.loss_type == 'upsample':
            one_hot_labels = torch.nn.functional.one_hot(
                batch['patch_56x56'].to(torch.int64), self.num_classes+1)
            # one_hot_labels = torch.nn.functional.one_hot(batch['patch_dino'].to(torch.int64), self.num_classes+1)
            total_loss = self.criterion(
                img_enc, one_hot_labels.to(torch.float32))

        if self.loss_type == 'ce':
            one_hot_labels = torch.nn.functional.one_hot(
                batch['indices_labels'].reshape(-1).to(torch.int64), self.num_classes+1)
            total_loss = self.criterion(
                img_enc, one_hot_labels.to(torch.float32))

        elif self.loss_type == 'contrastive':
            if self.weighted:
                loss = self.criterion(
                    img_enc, batch['indices_labels'].reshape(-1), batch['weighted_labels'].reshape(-1))
            else:
                loss = self.criterion(
                    img_enc, batch['indices_labels'].reshape(-1))

            total_loss = loss[0] + self.l1 * loss[1]
            # distance_matrix = loss[2]
            # aim_image = Image(distance_matrix)

            # self.aim_logger.experiment.track(aim_image, "Confusion Matrix")

        elif self.loss_type == 'both':
            one_hot_labels = torch.nn.functional.one_hot(
                batch['indices_labels'].reshape(-1).to(torch.int64), self.num_classes+1)
            cross_loss = self.crossentropyloss(
                img_enc, one_hot_labels.to(torch.float32))

            if self.weighted:
                loss = self.contrastiveloss(
                    img_enc, batch['indices_labels'].reshape(-1), batch['weighted_labels'].reshape(-1))
            else:
                loss = self.contrastiveloss(
                    img_enc, batch['indices_labels'].reshape(-1))

            total_loss = loss[0] + self.l1 * loss[1] + cross_loss

        self.log('train_loss', total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # print('######################')
        # print("loss type", self.loss_type)

        if self.read_embeds:
            img = batch['image']
        else:
            # maybe move it to dataset?
            img = torch.einsum('nhwc->nchw', batch['image'])
        img_enc = self.forward(img.float())

        if self.loss_type == 'upsample':
            one_hot_labels = torch.nn.functional.one_hot(
                batch['patch_56x56'].to(torch.int64), self.num_classes+1)
            # one_hot_labels = nn.functional.one_hot(batch['patch_dino'].to(torch.int64), self.num_classes+1)
            total_loss = self.criterion(
                img_enc, one_hot_labels.to(torch.float32))

        elif self.loss_type == 'ce':
            one_hot_labels = torch.nn.functional.one_hot(
                batch['indices_labels'].reshape(-1).to(torch.int64), self.num_classes+1)
            # print(img_enc.shape, one_hot_labels.shape)
            total_loss = self.criterion(
                img_enc, one_hot_labels.to(torch.float32))

        elif self.loss_type == 'contrastive':
            if self.weighted:
                loss = self.criterion(
                    img_enc, batch['indices_labels'].reshape(-1), batch['weighted_labels'].reshape(-1))
            else:
                loss = self.criterion(
                    img_enc, batch['indices_labels'].reshape(-1))

            total_loss = loss[0] + self.l1 * loss[1]
            # distance_matrix = loss[2]
            # aim_image = Image(distance_matrix)
            # self.aim_logger.experiment.track(aim_image, "Confusion Matrix")

            print(
                f'Iter: {batch_idx}, pos_loss: {loss[0].item()}, neg_loss = {self.l1} * {loss[1].item()}, loss: {total_loss.item()}')

        elif self.loss_type == 'both':
            one_hot_labels = torch.nn.functional.one_hot(
                batch['indices_labels'].reshape(-1).to(torch.int64), self.num_classes+1)
            cross_loss = self.crossentropyloss(
                img_enc, one_hot_labels.to(torch.float32))

            if self.weighted:
                print('weigtshdcn ', self.weighted)

                loss = self.contrastiveloss(
                    img_enc, batch['indices_labels'].reshape(-1), batch['weighted_labels'].reshape(-1))
            else:
                loss = self.contrastiveloss(
                    img_enc, batch['indices_labels'].reshape(-1))

            total_loss = loss[0] + self.l1 * loss[1] + cross_loss
            # distance_matrix = loss[2]
            # aim_image = Image(distance_matrix)

            # self.aim_logger.experiment.track(aim_image, "Confusion Matrix")
        # return total_loss
        return [img_enc.detach().cpu(), one_hot_labels.to(torch.float32).detach().cpu(), total_loss.detach().cpu().item()]

    def validation_epoch_end(self, outputs):
        preds = torch.cat([output[0] for output in outputs])
        preds = self.soft_activation(preds).reshape(-1, preds.shape[-1])
        # preds = preds
        labels = torch.cat([output[1] for output in outputs])
        labels = labels.argmax(dim=-1).reshape(-1)
        # labels = labels
        losses = torch.tensor([output[2] for output in outputs])

        roc = self.auc(preds.cuda(), labels.cuda()).cpu().detach()
        print(preds.device)
        preds = []
        labels = []
        loss = losses.mean()

        self.log('val_loss', loss)
        self.log('val_auc', roc)

        # file_path = os.path.join(
        #     f'./checkpoints/{self.experiment_name}', 'best_model.log')

        # with open(file_path, 'w') as f:
        #     f.write(
        #         f'best acc {self.checkpoint_callback_acc.best_model_path} {self.checkpoint_callback_acc.best_model_score}\n')
        #     f.write(
        #         f'best loss {self.checkpoint_callback_loss.best_model_path} {self.checkpoint_callback_loss.best_model_score}\n')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def fit_model(self):
        self.trainer.fit(model=self, train_dataloaders=self.train_dataloader,
                         val_dataloaders=self.val_dataloader)
        file_path = os.path.join(
            f'./checkpoints/{self.experiment_name}', 'best_model.log')

        with open(file_path, 'w') as f:
            f.write(
                f'best acc {self.checkpoint_callback_acc.best_model_path} {self.checkpoint_callback_acc.best_model_score}\n')
            f.write(
                f'best loss {self.checkpoint_callback_loss.best_model_path} {self.checkpoint_callback_loss.best_model_score}\n')
