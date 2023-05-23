import torch
import torchvision
import models_mae

from SimMIM.models.build import build_model
from SimMIM.config import get_config_from_file
from util.pos_embed import interpolate_pos_embed

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
              'chkpt_dir': '/home/hkhachatrian/mae/checkpoints/simmim_checkpoints/simmim_pretrain__vit_base__img224__800ep.pth',
              'config_path': '/home/hkhachatrian/mae/SimMIM/configs/vit_base__800ep/simmim_pretrain__vit_base__img224__800ep.yaml'},
    }
}


if __name__ == '__main__':
    model = get_model('mae')
    print(model)
