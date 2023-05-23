import argparse
from training_module import SegTrainer
import json
import pytorch_lightning as pl


def main(args):

    with open('default_hparams.json') as default_hparams:
        hparams = json.load(default_hparams)

    hparams['backbone_name'] = args.model
    hparams['emb_size'] = args.emb_size
    hparams['loss_type'] = args.loss_type
    hparams['bilinear_upsampling'] = args.bilinear_upsampling
    hparams['deconvolution'] = args.deconvolution
    hparams['backbone_freeze'] = args.backbone_freeze
    hparams['weighted'] = args.weighted
    hparams['experiment_name'] = args.experiment
    hparams['dataset_name'] = args.dataset_name
    hparams['annotation_train'] = args.annotation_train
    hparams['annotation_val'] = args.annotation_val
    hparams['resize_image'] = args.resize_image
    hparams['n_patches'] = args.n_patches
    hparams['read_embeds'] = args.read_embeds
    hparams['decoder_head'] = args.decoder_head
    hparams['train_embedds'] = args.train_embedds
    hparams['val_embedds'] = args.val_embedds
    hparams['train_embedds_labels'] = args.train_embedds_labels
    hparams['val_embedds_labels'] = args.val_embedds_labels

    pl.seed_everything(hparams['seed'])

    trainer = SegTrainer(hparams=argparse.Namespace(**hparams))
    trainer.fit_model()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Vit -> Segmentation task')

    parser.add_argument("--dataset_name", type=str,
                        help='Name of dataset one want to train', )
    parser.add_argument('--device', default='cuda', )
    parser.add_argument('--checkpoint', default='',
                        help='absolute path to checkpoint to be loaded', )
    parser.add_argument('--random_init', default=False, type=bool,)
    parser.add_argument('--annotation_train', default='')
    parser.add_argument('--annotation_val', default='')
    parser.add_argument('--train_embedds', default='')
    parser.add_argument('--val_embedds', default='')
    parser.add_argument('--train_embedds_labels', default='')
    parser.add_argument('--val_embedds_labels', default='')
    parser.add_argument('--experiment')
    parser.add_argument('--loss_type', default='contrastive')
    parser.add_argument('--weighted', default=False, type=bool)
    parser.add_argument('--backbone_freeze', default=False, type=bool)
    parser.add_argument('--bilinear_upsampling', action='store_true', )
    parser.add_argument('--deconvolution', action='store_true')
    parser.add_argument('--model', default='mae')
    parser.add_argument('--resize_image', action='store_false')
    parser.add_argument('--emb_size', default=768, type=int)
    parser.add_argument('--n_patches', default=14, type=int)
    parser.add_argument('--read_embeds', action='store_true')
    parser.add_argument('--decoder_head', default='')

    args = parser.parse_args()

    main(args)
