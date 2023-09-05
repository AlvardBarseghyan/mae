import numpy as np
from tqdm import tqdm

train_val = 'train'

model = "mae"
print('Loading files ...')
# embeds_ids = np.load(f'./varianceAnalysis/{model}_sorted_features.npy')
embeds_ids = np.load(f'./varianceAnalysis/{model}_sorted_features_ade20k.npy')
 
# # embeds = np.load(f'./embeddings/cs_patches_256/layers/{model}_cs4pc_256_{train_val}_7_embeds.npy')
# embeds = np.load(f'./embeddings/cs_full/mae_embeds_64_128_interp_{train_val}.npy')
embeds = np.load(f'./embeddings/ade20k_patches_224/mae_{train_val}_embeds.npy')

for i in [200]: # tqdm([10, 20, 30, 40, 50, 100, 200, 400, 600, 700, 750]): #[10, 20, 30, 40, 50]):
    tmp_embeds = np.delete(embeds, embeds_ids[:i], axis=1)
    saving_path = f'./embeddings/ade20k_patches_224/{model}{i}_{train_val}_embeds.npy'
    print(f"Saving with path {saving_path} ...")
    print(tmp_embeds.shape, embeds_ids.shape)
    np.save(saving_path, tmp_embeds)

# model = "mae"
# # degr = 'shift'
# print('Loading files ...')
# embeds_ids = np.load(f'./varianceAnalysis/{model}_sorted_features_f1m.npy')
# # embeds = np.load(f'./embeddings/fair1m/{model}_224_8shot_embeddings.npy')

# # for i in [200]: # tqdm([10, 20, 30, 40, 50, 100, 200, 400, 600, 700, 750]): #[10, 20, 30, 40, 50]):
# #     tmp_embeds = np.delete(embeds, embeds_ids[:i], axis=1)
# #     saving_path = f'./embeddings/fair1m/{model}{i}_224_8shot_embeddings.npy'
# #     print(f"Saving with path {saving_path} ...")
# #     print(tmp_embeds.shape, embeds_ids.shape)
# #     np.save(saving_path, tmp_embeds)

# degr = 'rotate'

# for j in [5, 10, 15, 20]:
#     embeds = np.load(f'./embeddings/fair1m/{model}_224_8shot_{degr}_{j}_embeddings.npy')

#     for i in [200]: # tqdm([10, 20, 30, 40, 50, 100, 200, 400, 600, 700, 750]): #[10, 20, 30, 40, 50]):
#         tmp_embeds = np.delete(embeds, embeds_ids[:i], axis=1)
#         saving_path = f'./embeddings/fair1m/{model}{i}_224_8shot_{degr}_{j}_embeddings.npy'
#         print(f"Saving with path {saving_path} ...")
#         print(tmp_embeds.shape, embeds_ids.shape)
#         np.save(saving_path, tmp_embeds)