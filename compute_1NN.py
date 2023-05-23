from get_eval_classes import get_eval_classes
import torch
import os
import numpy as np
from tqdm.notebook import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt

np.random.seed(32635)


def compute_1NN(path_to_knn, train_labels):
    k = 1
    top_k_classes = []

    name_to_chunk = {}
    for chunk_file in os.listdir(path_to_knn):
        chunk = int(chunk_file.split('_')[-1].split('.')[0])
        name_to_chunk[chunk] = chunk_file

    for _, knn_file in sorted(name_to_chunk.items()):
        knn = np.load(os.path.join(path_to_knn, knn_file),
                      allow_pickle=True).item()
#         knn = np.load(path_to_knn, allow_pickle=True).item()
        chunk_good_values = torch.tensor(knn['good_values'])
        chunk_good_indices = torch.tensor(knn['good_indices'])

        topk_indices = chunk_good_values.topk(k=k, largest=True).indices
        topk_labels = torch.zeros_like(topk_indices)

        j = 0
        for ind, good_index in zip(topk_indices, chunk_good_indices):
            real_index = good_index[ind]
            topk_labels[j] = train_labels[real_index]
            j += 1

        top_k_classes.append(topk_labels)
    top_k_classes = torch.cat(top_k_classes)

    return top_k_classes


def confusion_matrix(preds, gt, dataset_name, model, save_file_name, which_layer, split):
    path_to_matrixes = f'./knn_cm/{model}_{split}{which_layer}/'
    os.makedirs(path_to_matrixes, exist_ok=True)
    saving_path = os.path.join(path_to_matrixes, save_file_name + '.png')

    eval_labels, eval_label_names = get_eval_classes(dataset_name)
    acc_score = metrics.accuracy_score(gt, preds)
    cm = metrics.confusion_matrix(gt, preds, labels=eval_labels)
    plt.figure()
    plt.yticks(range(len(eval_labels)), eval_label_names)
    plt.xticks(range(len(eval_labels)), eval_label_names, rotation=90)
    plt.imshow(cm)
    plt.colorbar()
    plt.savefig(saving_path)

    return acc_score


def compute_predictions_and_store(annots, preds, prediction_path):
    predictions = preds.reshape(len(annots['images']), -1)

    for i, p in enumerate(predictions):
        annots['images'][i]['patch_labels_gt'] = annots['images'][i]['patch_labels']
        annots['images'][i]['patch_labels'] = p.detach().numpy()

    np.save(prediction_path, annots)
    print('predictions stored in', prediction_path)


def eval_predictions(model, path_to_knn, train_labels, val_labels, val_gt, dataset_name, tiles_count, split, which_layer, prediction_path):
    preds = compute_1NN(path_to_knn, train_labels)

    val_images_len = len(val_gt['images'])

    patch_count = preds.reshape(val_images_len, -1).shape[-1]

    train_images_count = len(train_labels) // patch_count // tiles_count
    val_images_count = len(val_labels) // patch_count // tiles_count

    acc_matrix_save_path = f'{split}_split_train_{train_images_count}_val_{val_images_count}'

    acc_score = confusion_matrix(
        preds, val_labels, dataset_name, model, acc_matrix_save_path, which_layer, split)

    with open("knn_accuray.txt", "a") as acc_file:
        acc_file.write(
            f"{model.upper()} layer{which_layer} {split}_split train:{train_images_count} val:{val_images_count} : {acc_score*100:.1f}\n")

    compute_predictions_and_store(val_gt, preds, prediction_path)
