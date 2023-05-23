import argparse
import time

import torch
import os
import numpy as np
from tqdm.notebook import tqdm

import h5py

from compute_1NN import eval_predictions

device = ''
np.random.seed(32635)


def cosine_distance_torch(x1, x2=None):
    x2 = x1 if x2 is None else x2
    return torch.mm(x1.to(float), x2.t().to(float))


def randomize_tensor(tensor):
    return tensor[torch.randperm(len(tensor))]


def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors
    y = x if type(y) == type(None) else y

    dist = cosine_distance_torch(x, y)
    return dist


class NN():
    def __init__(self, X=None, Y=None, p=2, from_hdf5=False):
        self.p = p
        self.train(X, Y, from_hdf5)

    def train(self, X, Y, from_hdf5=False):
        if from_hdf5:
            print('Loading Train Embeds')
            h5_file = h5py.File(X, 'r')
            self.train_pts = torch.tensor(np.array(h5_file['dataset_0']))
        else:
            print('Loading Train Embeds')
            self.train_pts = torch.from_numpy(
                np.load(X, allow_pickle=True)).to(device=device)
            # do next line only for mae model on full images
            # self.train_pts = self.train_pts.reshape(self.train_pts.shape[0] * self.train_pts.shape[1], self.train_pts.shape[2])

        print('Loading Train Labels')
        self.train_label = torch.from_numpy(np.load(Y, allow_pickle=True)).to(
            device=device, dtype=torch.int64)

    def __call__(self, x, from_hdf5=False, train_chunk_size=None, test_chunk_size=None):
        return self.predict(x, from_hdf5, train_chunk_size, test_chunk_size)

    def predict(self, path_to_data, from_hdf5=False, train_chunk_size=None, test_chunk_size=None):
        x = torch.from_numpy(
            np.load(path_to_data, allow_pickle=True)).to(device=device)
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(
                f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]


class KNN(NN):
    def __init__(self, X=None, Y=None, k=3, p=2, path_to_save='', from_hdf5=False):
        self.k = k
        self.save_path = path_to_save
        # self.hf = h5py.File(path_to_save, 'w')
        super().__init__(X, Y, p, from_hdf5)

    def train(self, X, Y, from_hdf5=False):
        super().train(X, Y, from_hdf5)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, path_to_data, from_hdf5=False, train_chunk_size=None, test_chunk_size=None):
        print('Loading samples for prediction')
        if from_hdf5:
            h5_file = h5py.File(path_to_data, 'r')
            x = torch.tensor(np.array(h5_file['dataset_0']))
        else:
            x = torch.from_numpy(
                np.load(path_to_data, allow_pickle=True)).to(device=device)
            # do next line only for mae model on full images
            # x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(
                f"{name} wasn't trained. Need to execute {name}.train() first")

        if train_chunk_size == None:
            train_chunk_size = len(self.train_pts)
        if test_chunk_size == None:
            test_chunk_size = len(x)

        print(test_chunk_size, train_chunk_size)

        test_iterator = int(len(x)/test_chunk_size) + \
            (len(x) % test_chunk_size > 0)
        train_iterator = int(len(self.train_pts)/train_chunk_size) + \
            (len(self.train_pts) % train_chunk_size > 0)

        for t in tqdm(range(test_iterator)):
            index_topn = []
            value_topn = []

            try:
                tmp = x[t*test_chunk_size:(t+1)*test_chunk_size]
                print('Computing distance')
                for i in tqdm(range(train_iterator)):
                    try:
                        dist = distance_matrix(
                            tmp, self.train_pts[i*train_chunk_size:(i+1)*train_chunk_size], self.p)
                    except IndexError:
                        dist = distance_matrix(
                            tmp, self.train_pts[i*train_chunk_size:], self.p)

                    knn = dist.topk(self.k, largest=True)  # .values
                    indices = knn.indices + i*train_chunk_size

                    index_topn.append(indices)
                    value_topn.append(knn.values)

                res = {"good_indices": np.concatenate(index_topn, axis=1),
                       "good_values":  np.concatenate(value_topn, axis=1)}
                chunk_path = self.save_path + \
                    f'{t*test_chunk_size}_{(t+1)*test_chunk_size}.npy'
                print('Storing...')
                print(chunk_path)
                np.save(chunk_path, res)
                # dict_group = self.hf.create_group(f'{t*size}_{(t+1)*size}')
                # for k, v in res.items():
                #     dict_group[k] = v

            except IndexError:
                tmp = x[t*test_chunk_size:]
                print('Computing distance')
                for i in tqdm(range(train_iterator)):
                    # dist = distance_matrix(x, self.train_pts, self.p)
                    try:
                        dist = distance_matrix(
                            tmp, self.train_pts[i*train_chunk_size:(i+1)*train_chunk_size], self.p)
                    except IndexError:
                        dist = distance_matrix(
                            tmp, self.train_pts[i*train_chunk_size:], self.p)

                    knn = dist.topk(self.k, largest=True)  # .values
                    indices = knn.indices + i*train_chunk_size

                    index_topn.append(indices)
                    value_topn.append(knn.values)

                res = {"good_indices": np.concatenate(index_topn, axis=1),
                       "good_values":  np.concatenate(value_topn, axis=1)}
                chunk_path = self.save_path + f'{t*test_chunk_size}_last.npy'
                print('Storing...')
                print(chunk_path)
                np.save(chunk_path, res)
                np.save(chunk_path, res)
                # dict_group = self.hf.create_group(f'{t*size_chunk}_last')
                # for k, v in res.items():
                #     dict_group[k] = v

        # self.hf.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='em')

    parser.add_argument('--model', default='mae')
    parser.add_argument("--dataset_name", type=str,
                        help='Name of dataset one want to train', )
    parser.add_argument('--device', default='cpu', )
    parser.add_argument('--train_embeds_path')
    parser.add_argument('--val_embeds_path')
    parser.add_argument('--val_annotation_path')
    parser.add_argument('--train_labels_path')
    parser.add_argument('--val_labels_path')
    parser.add_argument('--tiles_count', type=int, default=32)
    parser.add_argument('--split', default='4_initial')
    parser.add_argument('--which_layer', default='')
    parser.add_argument('--prediction_path')

    args = parser.parse_args()
    device = args.device
    model = args.model
    val_annotation_path = args.val_annotation_path
    dataset_name = args.dataset_name

    embeds_path = os.path.dirname(args.train_embeds_path)

    val_gt = np.load(val_annotation_path, allow_pickle=True).item()
    val_images_len = len(val_gt['images'])

    labels_train = torch.from_numpy(np.load(args.train_labels_path)).to(
        device=args.device, dtype=torch.int64)
    labels_val = torch.from_numpy(np.load(args.val_labels_path)).to(
        device=args.device, dtype=torch.int64)

    patch_count = labels_val.reshape(val_images_len, -1).shape[-1]

    number_of_train = labels_train.reshape(-1,
                                           patch_count).shape[0]//args.tiles_count
    number_of_val = labels_val.reshape(-1,
                                       patch_count).shape[0]//args.tiles_count

    print(embeds_path)
    path = os.path.join(
        embeds_path, f'predictions_knn/{args.model}_train_{number_of_train}_{args.split}_val_{number_of_val}')
    print(path)
    os.makedirs(path, exist_ok=True)
    path_to_save = os.path.join(
        path, f'{args.model}_train_{number_of_train}_{args.split}_val_{number_of_val}')
    print(path_to_save)
    # path_to_save = f'./annotations/cs_256_{model}_val_pred_knn.npy'
    print("Initing KNN...")
    knn = KNN(args.train_embeds_path, args.train_labels_path, k=10,
              p=2, path_to_save=path_to_save, from_hdf5=False)
    # specify train and test chunks. otherwise distance will compute across all samples
    start_ts = time.time()
    knn(args.val_embeds_path, train_chunk_size=10000, from_hdf5=False)
    end_ts = time.time()
    print(f'KNN computation lasts {(end_ts-start_ts)/60} minutes')
    eval_predictions(model, path, labels_train, labels_val, val_gt, dataset_name,
                     args.tiles_count, args.split, args.which_layer, args.prediction_path)
