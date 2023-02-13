import os 
import torch 
import argparse
import pytorch_lightning as pl

from loss import ContrastiveLoss
from dataset import MAEDataset
import models_mae

# TODO: integrate aim

class LightningMAE(pl.LightningModule):
    def __init__(self, model, l1=0.5, lr=1e-4, num_classes=5, margin=1):
        super().__init__()
        self.model_mae = model
        self.num_classes = num_classes
        self.l1 = l1
        self.lr = lr
        self.criterion = ContrastiveLoss(num_classes=num_classes, margin=margin)
        # self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):

        img = torch.einsum('nhwc->nchw', batch['image'])
        print("Getting image embeddings...")
        img_enc, mask, _ = self.model_mae.forward_encoder(img.float(), mask_ratio=0)
        img_enc = img_enc[:, 1:, :].reshape(-1, img_enc.shape[-1])
        print("Loss counting...")
        loss = self.criterion(img_enc, batch['indices_labels'].reshape(-1))
        total_loss = loss[0] + self.l1 * loss[1]
        self.log('train_loss', total_loss)
        print(f'Iter: {batch_idx}, pos_loss: {loss[0].item()}, neg_loss = {self.l1} * {loss[1].item()}, loss: {total_loss.item()}')

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

    args = parser.parse_args()


    if args.dataset_name.lower() == 'fair1m':
        root = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/train1000/'
        path_ann = os.path.join(root, 'few_shot_8.json')
        path_imgs = os.path.join(root, 'images')
        dataset = MAEDataset(path_ann, path_imgs, intersection_threshold=args.intersection_threshold, resize_image=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        num_classes = 5

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