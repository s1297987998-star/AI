import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):

    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(BCELoss, self).__init__()

        self.margin = margin

        self.reduce = reduce
        self.size_average = size_average

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, input, target):

        input, target = input.float(), target.float()

        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        positive_loss = self.BCEWithLogitsLoss(input, target)
        negative_loss = self.BCEWithLogitsLoss(-input, -target)

        loss = positive_mask * positive_loss + negative_mask * negative_loss

        if self.reduce:
            if self.size_average:
                return torch.mean(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(
                    positive_mask + negative_mask) != 0 else torch.mean(loss)
            return torch.sum(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(
                positive_mask + negative_mask) != 0 else torch.sum(loss)
        return loss


class ContrastiveLoss(nn.Module):
    """
    Document: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, batchSize, reduce=None, size_average=None):
        super(ContrastiveLoss, self).__init__()

        self.batchSize = batchSize
        self.concatIndex = self.getConcatIndex(batchSize)

        self.reduce = reduce
        self.size_average = size_average

        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-9)

    def forward(self, input, target):
        """
        Shape of input: (BatchSize, classNum, featureDim)
        Shape of target: (BatchSize, classNum), Value range of target: (-1, 0, 1)
        """

        target_ = target.detach().clone()
        target_[target_ != 1] = 0
        pos2posTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        pos2negTarget = 1 - pos2posTarget
        pos2negTarget[(target[self.concatIndex[0]] == 0) | (target[self.concatIndex[1]] == 0)] = 0
        pos2negTarget[(target[self.concatIndex[0]] == -1) & (target[self.concatIndex[1]] == -1)] = 0

        target_ = -1 * target.detach().clone()
        target_[target_ != 1] = 0
        neg2negTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        distance = self.cos(input[self.concatIndex[0]], input[self.concatIndex[1]])

        if self.reduce:
            pos2pos_loss = (1 - distance)[pos2posTarget == 1]
            pos2neg_loss = (1 + distance)[pos2negTarget == 1]
            neg2neg_loss = (1 + distance)[neg2negTarget == 1]

            if pos2pos_loss.size(0) != 0:
                if neg2neg_loss.size(0) != 0:
                    neg2neg_loss = torch.cat((torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[
                                                                                  :2 * pos2pos_loss.size(0)].cuda()),
                                              torch.sort(neg2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)
                if pos2neg_loss.size(0) != 0:
                    if pos2neg_loss.size(0) != 0:
                        pos2neg_loss = torch.cat((torch.index_select(pos2neg_loss, 0,
                                                                     torch.randperm(pos2neg_loss.size(0))[
                                                                     :2 * pos2pos_loss.size(0)].cuda()),
                                                  torch.sort(pos2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]),
                                                 0)

            loss = torch.cat((pos2pos_loss, pos2neg_loss, neg2neg_loss), 0)

            if self.size_average:
                return torch.mean(loss) if loss.size(0) != 0 else torch.mean(torch.zeros_like(loss).cuda())
            return torch.sum(loss) if loss.size(0) != 0 else torch.sum(torch.zeros_like(loss).cuda())

        return distance

    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum - 1):
            res[0] += [index for i in range(classNum - index - 1)]
            res[1] += [i for i in range(index + 1, classNum)]
        return res


class PartialSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(PartialSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, targets):
        """
        Args:
            features: (BatchSize, ClassNum, ProjDim) - 经过投影和归一化的特征
            targets: (BatchSize, ClassNum) - 标签 {-1, 0, 1}
        Returns:
            Scalar Loss
        """
        device = features.device
        batch_size = features.shape[0]
        class_num = features.shape[1]

        loss = 0.0
        valid_classes = 0  # 记录有多少个类别产生了有效的 Loss

        # 针对每个类别单独计算 SupCon Loss，然后求平均
        for c in range(class_num):
            # 1. 提取当前类别的特征 (Batch, Dim) 和标签 (Batch,)
            feat_c = features[:, c, :]
            label_c = targets[:, c]

            # 2. 构建掩码
            # Anchor Mask: 只有标签为 1 的样本才能作为锚点计算 Loss
            anchor_mask = (label_c == 1).float()

            # Positive Mask: 用于寻找同类 (标签也是 1 的其他样本)
            # mask[i] = 1 if label[i] == 1
            pos_mask = (label_c == 1).float()

            # Valid Mask: 标签已知 (1 或 -1) 的样本参与分母计算，0 忽略
            valid_mask = (label_c != 0).float()

            # 如果当前 Batch 里该类别没有正样本，或者正样本少于2个(无法配对)，跳过
            if anchor_mask.sum() < 2:
                continue

            # 3. 计算相似度矩阵 (Batch, Batch)
            # feat_c 已经是归一化的，所以 dot product 就是 cosine similarity
            sim_matrix = torch.matmul(feat_c, feat_c.T) / self.temperature

            # 数值稳定性：减去每行的最大值
            sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_max.detach()

            # 4. 计算分母 (Denominator)
            # 只有 valid (1或-1) 的样本才进入分母，且不能包含自己 (diag)
            exp_sim = torch.exp(sim_matrix)
            mask_valid = valid_mask.unsqueeze(0) * valid_mask.unsqueeze(1)  # (B, B)
            mask_diag = 1 - torch.eye(batch_size).to(device)

            # 分母 = sum(exp(sim(i, k))) for k in valid and k != i
            denominator = (exp_sim * mask_valid * mask_diag).sum(dim=1)

            # 5. 计算分子 (Numerator) & Log Probability
            # log(exp(sim(i, j)) / denominator) = sim(i, j) - log(denominator)
            log_prob = sim_matrix - torch.log(denominator + 1e-7).unsqueeze(1)

            # 6. 计算 Loss
            # 只有 (i, j) 都是正样本时，才计算这一项
            mask_pos_pair = pos_mask.unsqueeze(0) * pos_mask.unsqueeze(1) * mask_diag

            # 计算每个 Anchor i 的平均 log_prob
            # sum over j, divide by number of positives for i
            mean_log_prob = (mask_pos_pair * log_prob).sum(dim=1) / (mask_pos_pair.sum(dim=1) + 1e-7)

            # 只对本身是 Anchor (label=1) 的样本求和
            loss_c = - (mean_log_prob * anchor_mask).sum() / (anchor_mask.sum() + 1e-7)

            loss += loss_c
            valid_classes += 1

        if valid_classes == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss / valid_classes
