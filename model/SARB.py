import numpy as np

import torch
import torch.nn as nn

from sklearn.mixture import GaussianMixture
from .backbone.resnet import resnet101
from .GraphNeuralNetwork import GatedGNN
from .SemanticDecoupling import SemanticDecoupling
from .Element_Wise_Layer import Element_Wise_Layer


class SARB(nn.Module):

    def __init__(self, adjacencyMatrix, wordFeatures, prototypeNum,
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80, wordFeatureDim=300, timeStep=3,
                 isAlphaLearnable=True, isBetaLearnable=True):

        super(SARB, self).__init__()

        self.backbone = resnet101()

        if imageFeatureDim != 2048:
            self.changeChannel = nn.Sequential(nn.Conv2d(2048, imageFeatureDim, kernel_size=1, stride=1, bias=False),
                                               nn.BatchNorm2d(imageFeatureDim), )

        self.classNum = classNum
        self.prototypeNum = prototypeNum

        self.timeStep = timeStep
        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.wordFeatureDim = wordFeatureDim
        self.imageFeatureDim = imageFeatureDim

        self.wordFeatures = self.load_features(wordFeatures)
        self.inMatrix, self.outMatrix = self.load_matrix(adjacencyMatrix)

        self.SemanticDecoupling = SemanticDecoupling(classNum, imageFeatureDim, wordFeatureDim,
                                                     intermediaDim=intermediaDim)
        self.GraphNeuralNetwork = GatedGNN(imageFeatureDim, timeStep, self.inMatrix, self.outMatrix)

        self.fc = nn.Linear(2 * imageFeatureDim, outputDim)
        self.classifiers = Element_Wise_Layer(classNum, outputDim)

        self.cos = torch.nn.CosineSimilarity(dim=3, eps=1e-9)
        self.prototype = []

        self.alpha = nn.Parameter(torch.tensor(0.5).float(), requires_grad=isAlphaLearnable)
        self.beta = nn.Parameter(torch.tensor(0.5).float(), requires_grad=isBetaLearnable)

        # ------------------------------------------------------------------
        # 投影头：将特征映射到对比学习空间 (通常比特征维度低，如 128)
        self.contrastive_head = nn.Sequential(
            nn.Linear(imageFeatureDim, imageFeatureDim),
            nn.ReLU(inplace=True),
            nn.Linear(imageFeatureDim, 128)  # 128是对比特征维度
        )
        # ------------------------------------------------------------------

    def forward(self, input, target=None):

        batchSize = input.size(0)

        featureMap = self.backbone(input)  # (batchSize, channel, imgSize, imgSize)
        if featureMap.size(1) != self.imageFeatureDim:
            featureMap = self.changeChannel(featureMap)  # (batchSize, imgFeatureDim, imgSize, imgSize)

        semanticFeature = self.SemanticDecoupling(featureMap, self.wordFeatures)[0]  # (batchSize, classNum, outputDim)

        # -----------------------------------------------------------------
        # 计算用于对比学习的归一化特征
        # 将维度展平以便通过全连接层: (batchSize * classNum, imageFeatureDim)
        flat_semantic = semanticFeature.view(-1, self.imageFeatureDim)
        proj_features = self.contrastive_head(flat_semantic)
        # 归一化是对比学习的关键
        proj_features = torch.nn.functional.normalize(proj_features, dim=1)

        # 恢复形状: (batchSize, classNum, 128)
        contrastive_features = proj_features.view(batchSize, self.classNum, -1)
        # -----------------------------------------------------------------

        # Predict Category
        feature = self.GraphNeuralNetwork(semanticFeature)
        output = torch.tanh(self.fc(torch.cat((feature.view(batchSize * self.classNum, -1),
                                               semanticFeature.view(-1, self.imageFeatureDim)), 1)))
        output = output.contiguous().view(batchSize, self.classNum, self.outputDim)
        result = self.classifiers(output)  # (batchSize, classNum)

        if not self.training:
            return result

        if target is None:
            return result, semanticFeature
        # -------------------------------------------------------------------
        # 检查原型是否已就绪。在 mixupEpoch 之前，self.prototype 是空列表 []
        # 如果是列表，说明还没有运行 computePrototype，不能进行 Mixup
        if isinstance(self.prototype, list):
            # 返回: result, semanticFeature, mixedResult1, mixedTarget1, mixedResult2, mixedTarget2, contrastive_features
            # 中间的 Mixup 结果全部填 None
            return result, semanticFeature, None, None, None, None, contrastive_features
        # -------------------------------------------------------------------

        self.alpha.data.clamp_(min=0, max=1)
        self.beta.data.clamp_(min=0, max=1)

        # Instance-level Mixup
        coef, mixedTarget_1 = self.mixupLabel(target, torch.flip(target, dims=[0]), self.alpha)
        coef = coef.unsqueeze(-1).repeat(1, 1, self.outputDim)
        mixedSemanticFeature_1 = coef * semanticFeature + (1 - coef) * torch.flip(semanticFeature, dims=[0])

        # Predict Category
        mixedFeature_1 = self.GraphNeuralNetwork(mixedSemanticFeature_1)
        mixedOutput_1 = torch.tanh(self.fc(torch.cat((mixedFeature_1.view(batchSize * self.classNum, -1),
                                                      mixedSemanticFeature_1.view(-1, self.imageFeatureDim)), 1)))
        mixedOutput_1 = mixedOutput_1.contiguous().view(batchSize, self.classNum, self.outputDim)
        mixedResult_1 = self.classifiers(mixedOutput_1)  # (batchSize, classNum)

        # Prototype-level Mixup
        prototype = self.prototype[:, torch.randint(self.prototype.size(1), (1,)), :].squeeze()
        prototype = prototype.unsqueeze(0).repeat(batchSize, 1, 1)

        mask = torch.rand(target.size()).cuda()
        mask = mask * (target == 0)
        mask[torch.arange(target.size(0)), torch.argmax(mask, dim=1)] = 1
        mask[mask != 1] = 0

        mixedSemanticFeature_2 = self.beta * semanticFeature + (1 - self.beta) * prototype
        mixedSemanticFeature_2 = mask.unsqueeze(-1).repeat(1, 1, self.outputDim) * mixedSemanticFeature_2 + \
                                 (1 - mask).unsqueeze(-1).repeat(1, 1, self.outputDim) * semanticFeature
        mixedTarget_2 = (1 - mask) * target + mask * (1 - self.beta)

        # Predict Category
        mixedFeature_2 = self.GraphNeuralNetwork(mixedSemanticFeature_2)
        mixedOutput_2 = torch.tanh(self.fc(torch.cat((mixedFeature_2.view(batchSize * self.classNum, -1),
                                                      mixedSemanticFeature_2.view(-1, self.imageFeatureDim)), 1)))
        mixedOutput_2 = mixedOutput_2.contiguous().view(batchSize, self.classNum, self.outputDim)
        mixedResult_2 = self.classifiers(mixedOutput_2)  # (batchSize, classNum)

        # --------------------------------------------------------------------
        return result, semanticFeature, mixedResult_1, mixedTarget_1, mixedResult_2, mixedTarget_2, contrastive_features

    def mixupLabel(self, label1, label2, alpha):

        matrix = torch.ones_like(label1).cuda()
        matrix[(label1 == 0) & (label2 == 1)] = alpha

        return matrix, matrix * label1 + (1 - matrix) * label2

    def computePrototype(self, train_loader):

        from sklearn.mixture import GaussianMixture

        self.eval()
        prototypes, features = [], [torch.zeros(10, self.outputDim) for i in range(self.classNum)]

        for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(train_loader):

            input, target, groundTruth = input.cuda(), target.float().cuda(), groundTruth.cuda()

            with torch.no_grad():
                featureMap = self.backbone(input)  # (batchSize, channel, imgSize, imgSize)
                if featureMap.size(1) != self.imageFeatureDim:
                    featureMap = self.changeChannel(featureMap)  # (batchSize, imgFeatureDim, imgSize, imgSize)

                semanticFeature = self.SemanticDecoupling(featureMap, self.wordFeatures)[
                    0]  # (batchSize, classNum, outputDim)

                feature = semanticFeature.cpu()

                for i in range(self.classNum):
                    target_cpu = target.cpu()  # 将整个 target 移回 CPU
                    mask = (target_cpu[:, i] == 1)  # 生成 CPU 上的 mask
                    if mask.sum() > 0:
                        features[i] = torch.cat((features[i], feature[mask, i]), dim=0)

            # --- 聚类/拟合部分修改 ---
        for i in range(self.classNum):
            # 获取有效数据（跳过初始化的前10个全0向量）
            valid_features = features[i][10:].numpy()
            n_samples = valid_features.shape[0]

            # 初始化当前类的原型容器
            cluster_means = None

            # ================= [策略 1: 处理极少样本] =================
            # 如果没有样本，用全0填充 (极端情况)
            if n_samples == 0:
                cluster_means = np.zeros((self.prototypeNum, self.outputDim))

            # 如果样本数 < 原型数 (比如只有 5 个样本，要 10 个原型)
            # 解决方案：最大只能拟合 n_samples 个高斯分布
            elif n_samples < self.prototypeNum:
                # 尝试拟合尽可能多的高斯分布 (最大为 n_samples)
                # 使用 'spherical' 或 'diag' 以增加稳定性
                try:
                    gmm = GaussianMixture(n_components=n_samples,
                                          covariance_type='diag',
                                          reg_covar=1e-4,  # 较大的正则化
                                          random_state=0)
                    gmm.fit(valid_features)
                    means = gmm.means_  # shape: (n_samples, dim)
                except:
                    # 如果连这也失败了，直接用样本本身作为均值
                    means = valid_features

                # 补齐到 prototypeNum 个 (循环填充)
                # 例如：有3个均值 [A, B, C]，需要10个。
                # 结果：[A, B, C, A, B, C, A, B, C, A]
                repeat_times = (self.prototypeNum // means.shape[0]) + 1
                cluster_means = np.tile(means, (repeat_times, 1))[:self.prototypeNum, :]

            # ================= [策略 2: 样本充足，自适应拟合] =================
            else:
                # 尝试循环增加 reg_covar 直到成功
                # 这是一个标准的处理矩阵奇异的工程方法，不改变算法本质
                success = False
                reg_covars = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]

                for reg in reg_covars:
                    try:
                        gmm = GaussianMixture(n_components=self.prototypeNum,
                                              covariance_type='diag',
                                              reg_covar=reg,
                                              max_iter=100,
                                              random_state=0)
                        gmm.fit(valid_features)
                        cluster_means = gmm.means_
                        success = True
                        break  # 成功了就跳出循环
                    except Exception:
                        continue  # 失败了就增加 reg_covar 重试

                # 如果所有 reg_covar 都失败 (几乎不可能发生，除非数据全是 NaN)
                if not success:
                    # 最后的保底：视为单高斯分布 (n=1) 然后复制
                    # 这依然是 GMM (K=1 的特例)
                    try:
                        gmm = GaussianMixture(n_components=1, covariance_type='diag', reg_covar=1.0).fit(valid_features)
                        mean = gmm.means_
                        cluster_means = np.repeat(mean, self.prototypeNum, axis=0)
                    except:
                        # 彻底崩溃 (数据有问题)，用算术平均
                        mean = np.mean(valid_features, axis=0, keepdims=True)
                        cluster_means = np.repeat(mean, self.prototypeNum, axis=0)

            # 转换为 Tensor
            prototypes.append(torch.tensor(cluster_means, dtype=torch.float32).cuda())

        self.prototype = torch.stack(prototypes, dim=0)

    def load_features(self, wordFeatures):
        return nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)

    def load_matrix(self, mat):
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix, _out_matrix = nn.Parameter(torch.from_numpy(_in_matrix), requires_grad=False), nn.Parameter(
            torch.from_numpy(_out_matrix), requires_grad=False)
        return _in_matrix, _out_matrix

# =============================================================================
# Help Functions
# =============================================================================
