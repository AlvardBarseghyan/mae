import os
import cv2
import torch
import seaborn as sns
import numpy as np
import models_mae
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from util.pos_embed import interpolate_pos_embed
from tqdm import tqdm

device='cpu'
img_size = (1024, 2048)
# arch='mae_vit_large_patch16'
arch='mae_vit_base_patch16'
size_to = (64, 128)
model_mae = getattr(models_mae, arch)()

# chkpt_dir = '/mnt/lwll/lwll-coral/hrant/mae_checkpoints/mae_checkpoints/mae_visualize_vit_large_ganloss.pth'
chkpt_dir = '/mnt/lwll/lwll-coral/hrant/mae_checkpoints/mae_checkpoints/mae_pretrain_vit_base.pth'

checkpoint = torch.load(chkpt_dir, map_location='cpu')

state_dict = checkpoint['model']
##### Encoder position embeddings

# interpolate_pos_embed(model_mae, state_dict)

n, c, hw = state_dict['pos_embed'].transpose(1, 2).shape
pos_embed_weight = state_dict['pos_embed'][:, 1:].transpose(1, 2)

n, c, hw = pos_embed_weight.shape
h = w = int(hw ** 0.5)
pos_embed_weight = pos_embed_weight.reshape(n, c, h, w)
print("embedding shape:", pos_embed_weight.shape)
pos_embed_weight = F.interpolate(pos_embed_weight, size=size_to, mode='bicubic')
print(pos_embed_weight.shape)
pos_embed_weight = pos_embed_weight.reshape(n, c, -1).permute(0, 2, 1)
print(pos_embed_weight.shape)
cls_token_weight = state_dict['pos_embed'][:, 0].unsqueeze(1)
state_dict['pos_embed'] = torch.cat((cls_token_weight, pos_embed_weight), dim=1)

##### Decoder position embeddings

# n, c, hw = state_dict['decoder_pos_embed'].transpose(1, 2).shape
# pos_embed_weight = state_dict['decoder_pos_embed'][:, 1:].transpose(1, 2)

# n, c, hw = pos_embed_weight.shape
# h = w = int(hw ** 0.5)
# pos_embed_weight = pos_embed_weight.reshape(n, c, h, w)

# pos_embed_weight = F.interpolate(pos_embed_weight, size=(16, 16), mode='bicubic')

# pos_embed_weight = pos_embed_weight.reshape(n, c, -1).permute(0, 2, 1)

# cls_token_weight = state_dict['decoder_pos_embed'][:, 0].unsqueeze(1)

# state_dict['decoder_pos_embed'] = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
###############################

msg = model_mae.load_state_dict(state_dict, strict=False)
print(msg)
model_mae.to(device)

########################################
############# Annotations ##############

embeds = []
labels = []
# annot = np.load('./annotations/cs/cs4pc_256_train.npy', allow_pickle=True).item()
annot = np.load('./annotations/cs/cs_val.npy', allow_pickle=True).item()

for ann in tqdm(annot['images'], total=len(annot['images'])):
    # file = ann['file_name'].replace('/cs4pc_256/', '/vision/images/cs/cs4pc_256/')
    file = ann['file_name'].replace('/leftImg8bit/', '/vision/images/cs/full_ds/')
    # print(file)
    img = cv2.imread(file)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    x, _, ids_restore = model_mae.forward_encoder(img.to(device), mask_ratio=0)

    x_ = x[:, 1:, :]  # no cls token
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

    img_enc = x[:, 1:, :].detach().numpy()
    embeds.append(img_enc.squeeze(0))
    black = cv2.resize(ann['black_image'], (128, 64), interpolation=cv2.INTER_NEAREST_EXACT)
    labels.append(black.reshape(-1))

embeds = np.concatenate(embeds, axis=0)
labels = np.concatenate(labels)

print(embeds.shape, labels.shape)
np.save('./embeddings/cs_full/mae_embeds_64_128_interp_val.npy', embeds)
np.save('./embeddings/cs_full/mae_labels_interp_val.npy', labels)

# np.save('./embeddings/cs_patches_256/mae_embeds_256interp_train.npy', embeds)
# np.save('./embeddings/cs_patches_256/mae_labels_train.npy', labels)
