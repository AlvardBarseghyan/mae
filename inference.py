from modeling import Segmenter
import os
import json
import torch
import torch.nn as nn
import argparse
from compute_1NN import compute_predictions_and_store
from tqdm import tqdm
import numpy as np


def load_model_from_checkpoint(config_path, checkpoint_path, device):
    with open(config_path) as config:
        hparams = json.load(config)
    backbone_name = hparams['backbone_name']
    num_classes = hparams['num_classes']
    classifier = hparams['decoder_head']
    backbone_freeze = hparams['backbone_freeze']
    read_embeds = hparams['read_embeds']
    emb_size = hparams['emb_size']

    model = Segmenter(backbone_name=backbone_name, num_classes=num_classes, emb_size=emb_size, 
                      backbone_freeze=backbone_freeze, classifier_name=classifier, read_embeds=read_embeds)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    return model.to(device)


def load(val_annotation):
    try:
        tmp = np.load(val_annotation, allow_pickle=True).item()['images']
        return np.load(val_annotation, allow_pickle=True).item()
    except:
        return np.load(val_annotation, allow_pickle=True)


def inference_model(config_path, checkpoint_path, val_embeds, device):
    model = load_model_from_checkpoint(config_path, checkpoint_path, device)
    softmax = nn.Softmax(dim=-1)
    model.eval()
    predictiions = []
    with torch.no_grad():
        for embeds in tqdm(val_embeds):
            embeds = torch.from_numpy(embeds).unsqueeze(0)
            preds = model(embeds.to(device))
            preds = softmax(preds)
            predictiions.append(preds.cpu().detach().argmax(dim=-1))

    predictiions = torch.cat(predictiions, dim=0)

    return predictiions


def get_best_checkpoint(experiment_name, best='acc'):
    ckpt_path = './checkpoints/'
    ckpt_name = os.path.join(ckpt_path, experiment_name)
    print("############", os.path.join(ckpt_name, 'best_model.log'))
    if os.path.exists(os.path.join(ckpt_name, 'best_model.log')):
        with open(os.path.join(ckpt_name, 'best_model.log')) as f:
            data = f.readlines()

        for i in data:
            if best in i:
                print("***************")
                print(i.strip().split()[2])
                return i.strip().split()[2], os.path.join(ckpt_name, 'config.json')

    with open(os.path.join(ckpt_path, 'best_checkpoints_new.json')) as f:
        data = json.load(f)

    key = 'val_auc' if best == 'acc' else 'val_loss'
    # epoch = data[experiment_name][key]
    tmp = data.get(experiment_name, None)
    if tmp:
        epoch = tmp[key]
        if os.path.exists(os.path.join(ckpt_name, f'val_loss_epoch={epoch}.ckpt')):
            return os.path.join(ckpt_name, f'val_loss_epoch={epoch}.ckpt'), os.path.join(ckpt_name, 'config.json')
        else:
            pass
            # epoch = 
    epoch = 219
    print("Epoch_219")
    return os.path.join(ckpt_name, f'val_loss_epoch={epoch}.ckpt'), os.path.join(ckpt_name, 'config.json')



if __name__ == '__main__':
    root = './embeddings/cs_patches_256/'
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        '--experiment_name', default='cs4pcs_256_dino_noise_4_layer_12_linear', type=str)
    parser.add_argument('--val_annotation',
                        default='./annotations/cs/cs4pc_256_val.npy', type=str)
    parser.add_argument('--val_embeds_path', default=root +
                        'layers_noise_4/dino_val_12_embeds.npy', type=str)
    parser.add_argument('--val_labels_path', default=root +
                        'layers_noise_4/dino_val_12_labels.npy', type=str)
    parser.add_argument('--prediction_path',
                        default='./predictions/cropped/eval_dino_cs_patches_256_linear_noise_4_layer_12.npy',
                        type=str)
    # parser.add_argument('--emb_size', default=768, type=int)
    
    parser.add_argument('--best', default='acc',
                        type=str)  # loss for loss type
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    print(args.experiment_name)
    checkpoint, config_path = get_best_checkpoint(
        args.experiment_name, best=args.best)
    val_annotation = load(args.val_annotation)
    val_embeds = load(args.val_embeds_path)
    val_labels = load(args.val_labels_path)
    val_preds = inference_model(
        config_path, checkpoint, val_embeds, args.device)
    print("prediction shape:", val_preds.shape)
    print(args.prediction_path)


    compute_predictions_and_store(
        annots=val_annotation, preds=val_preds, prediction_path=args.prediction_path)
