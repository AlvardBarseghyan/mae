import torch
import torchvision
import models_mae
from ibot import models
from transformers import AutoImageProcessor, BeitModel

from SimMIM.models.build import build_model
from SimMIM.config import get_config_from_file
from util.pos_embed import interpolate_pos_embed

from ijepa.src.helper import load_checkpoint, init_model, init_opt

from collections import OrderedDict


def get_model_config(model_name, model_version='B'):
    return MODEL_CONFIGS[model_name][model_version]


def get_model(model_name, model_version='B', device='cuda', img_size=224):
    model_config = get_model_config(model_name, model_version)

    if model_name == 'dino':
        model = torch.hub.load(
            'facebookresearch/dino:main', model_config['arch'])

    elif model_name == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', model_config['arch'])

    elif model_name == 'mae':
        model = getattr(models_mae, model_config['arch'])(img_size=img_size)
        checkpoint = torch.load(model_config['chkpt_dir'], map_location=device)
        checkpoint_model = checkpoint['model']
        if img_size != 224:
            interpolate_pos_embed(model, checkpoint_model)
        model.load_state_dict(checkpoint_model, strict=False)

    elif model_name == 'sup_vit':
        model = getattr(torchvision.models, model_config['arch'])(
            weights='IMAGENET1K_V1')
        
    elif model_name == 'simmim':
        cfg_file = model_config['config_path']
        config = get_config_from_file(cfg_file)

        model = build_model(config)
        checkpoint = torch.load(model_config['chkpt_dir'])
        model.load_state_dict(checkpoint['model'])

    elif model_name == 'ibot':
        model = models.__dict__[model_config['arch']](
            patch_size=16,
            return_all_tokens=True,
        )
        checkpoint = torch.load(model_config['chkpt_dir'])
        model.load_state_dict(checkpoint['state_dict'])

    elif model_name == 'ijepa':
        model, predictor = init_model(
        device=device,
        patch_size=model_config['patch'],
        crop_size=model_config['image_size'],
        model_name=model_config['arch']
        )
        checkpoint = torch.load(model_config['chkpt_dir'])
        epoch = checkpoint['epoch']
        pretrained_dict = checkpoint['encoder']
        new_weights = OrderedDict()
        for key in pretrained_dict.keys():
            new_key = key.replace('module.', '')
            new_weights[new_key] = pretrained_dict[key]

        msg = model.load_state_dict(new_weights)

        print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    elif model_name == 'beit':
        # from transformers import BeitModel
        model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")


    else:
        raise ValueError(f"Unknown model {model_name}")

    print(f"{model_config['arch']} model is loaded!")
    return model.to(device)


MODEL_CONFIGS = {
    'dino': {
        'B': {'arch': 'dino_vitb16', 'patch': 16, 'emb_size': 768, 'layers': 12},
    },
    'dinov2': {
        'B': {'arch': 'dinov2_vitb14', 'patch': 14, 'emb_size': 768, 'layers': 12},
        'L': {'arch': 'dinov2_vitl14', 'patch': 14, 'emb_size': 1024, 'layers': 24}
    },
    'mae': {
        'B': {'arch': 'mae_vit_base_patch16', 'patch': 16, 'emb_size': 768, 'layers': 12,
              'chkpt_dir': '/mnt/lwll/lwll-coral/hrant/mae_checkpoints/mae_checkpoints/mae_pretrain_vit_base.pth'},
        'L': {'arch': 'mae_vit_large_patch16', 'patch': 16, 'emb_size': 1024, 'layers': 24,
              'chkpt_dir': '/mnt/lwll/lwll-coral/hrant/mae_checkpoints/mae_checkpoints/mae_pretrain_vit_large.pth'},
        'L_GAN_LOSS': {'arch': 'mae_vit_large_patch16', 'patch': 16, 'emb_size': 1024, 'layers': 24,
                       'chkpt_dir': '/mnt/lwll/lwll-coral/hrant/mae_checkpoints/mae_checkpoints/mae_visualize_vit_large_ganloss.pth'}
    },
    'sup_vit': {
        'B': {'arch': 'vit_b_16', 'patch': 16, 'emb_size': 768, 'layers': 12},
        'L': {'arch': 'vit_l_16', 'patch': 16, 'emb_size': 1024, 'layers': 24}
    },
    'simmim': {
        'B': {'arch': 'simmim_vit_base__800ep', 'patch': 16, 'emb_size': 768, 'layers': 12, 
              'chkpt_dir': '/mnt/lwll/lwll-coral/hrant/mae/checkpoints/simmim_checkpoints/simmim_pretrain__vit_base__img224__800ep.pth',
              'config_path': '/mnt/lwll/lwll-coral/hrant/mae/SimMIM/configs/vit_base__800ep/simmim_pretrain__vit_base__img224__800ep.yaml'},
    },
    'ibot': {
        'B': {'arch': 'vit_base', 'patch': 16, 'emb_size': 768, 'layers': 12, 
              'chkpt_dir': '/mnt/lwll/lwll-coral/hrant/mae/checkpoints/ibot_checkpoints/checkpoint_teacher.pth'
        },
    },
    'ijepa': {
        'B': {'arch': 'vit_huge', 'patch': 14, 'emb_size': 1280, 'layers': 32, 'image_size': 224,
              'chkpt_dir': '/mnt/lwll/lwll-coral/hrant/mae/ijepa/IN1K-vit.h.14-300e.pth'
        },
    },
    'beit': {
        'B': {'arch': 'beit', 'patch': 16, 'emb_size': 768, 'layers': 12},
    },
}


if __name__ == '__main__':
    model = get_model('mae', img_size=(256,512))
    print(model)
