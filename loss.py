import torch
import torch.nn as nn
# import torchmetrics


def cosine_distance_torch(x1, x2=None):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t())


class ContrastiveLoss(nn.Module):
    def __init__(self, num_classes=5, margin=1.0) -> None:
        super().__init__()
        self.num_classes=num_classes
        self.margin = margin

    def forward(self, img_enc_1, labels, img_enc_2=None):
        if not img_enc_2:
            # cos_dist = 1 - torchmetrics.functional.pairwise_cosine_similarity(img_enc_1)
            cos_dist = cosine_distance_torch(img_enc_1)
        else:
            # cos_dist = 1 - torchmetrics.functional.pairwise_cosine_similarity(img_enc_1, img_enc_2)
            cos_dist = cosine_distance_torch(img_enc_1, img_enc_2)

        # d = 0 means y1 and y2 are supposed to be same
        # d = 1 means y1 and y2 are supposed to be different

        distance_matrix = (labels.repeat(labels.shape[0], 1) - labels.repeat(labels.shape[0], 1).T)
        distance_matrix = distance_matrix.abs().sign()


        positive_loss = (1 - distance_matrix) * cos_dist
        if (1 - distance_matrix).sum() != 0:
            positive_loss /= (1 - distance_matrix).sum()
        # positive_loss = torch.nan_to_num(positive_loss)

        delta = self.margin - cos_dist # if margin == 1, then 1 - cos_dist == cos_sim
        delta= torch.clamp(delta, min=0.0, max=None)
        negative_loss = distance_matrix * delta
        if distance_matrix.sum() != 0:
            negative_loss /= distance_matrix.sum()
        # negative_loss = torch.nan_to_num(negative_loss)

        agg_loss = torch.zeros((self.num_classes+1, self.num_classes+1))
        agg_d = torch.zeros((self.num_classes+1, self.num_classes+1))
        label_masks = [labels==i for i in range(self.num_classes+1)]
        for i in range(self.num_classes + 1):
            for j in range(self.num_classes + 1):
                # print(distance_matrix[label_masks[i]][:, label_masks[j]])
                agg_loss[i][j] = cos_dist[label_masks[i]][:, label_masks[j]].mean()
                # agg_d[i][j] = distance_matrix[label_masks[i]][:, label_masks[j]].mean()

        print(*[x.sum().item() for x in label_masks])
        print(agg_loss)
        # print(agg_d)

        return positive_loss.sum(), negative_loss.sum()