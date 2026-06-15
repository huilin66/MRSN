# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class FocalLoss(nn.Layer):
    """
    Focal Loss.

    Code referenced from:
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py

    Args:
        gamma (float): the coefficient of Focal Loss.
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, gamma=2.0, ignore_index=255, edge_label=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.edge_label = edge_label

    def forward(self, logit, label):
        logit = paddle.reshape(
            logit, [logit.shape[0], logit.shape[1], -1])  # N,C,H,W => N,C,H*W
        logit = paddle.transpose(logit, [0, 2, 1])  # N,C,H*W => N,H*W,C
        logit = paddle.reshape(logit,
                               [-1, logit.shape[2]])  # N,H*W,C => N*H*W,C
        label = paddle.reshape(label, [-1, 1])
        range_ = paddle.arange(0, label.shape[0])
        range_ = paddle.unsqueeze(range_, axis=-1)
        label = paddle.cast(label, dtype='int64')
        label = paddle.concat([range_, label], axis=-1)
        logpt = F.log_softmax(logit)
        logpt = paddle.gather_nd(logpt, label)

        pt = paddle.exp(logpt.detach())
        loss = -1 * (1 - pt)**self.gamma * logpt
        loss = paddle.mean(loss)
        return loss


@manager.LOSSES.add_component
class FocalLoss_Smooth(nn.Layer):
    """
    Focal Loss with Label Smoothing.
    
    基于现有 FocalLoss 扩展，添加标签平滑功能。
    标签平滑能缓解过拟合，与 Focal Loss 的难样本聚焦互补。

    Args:
        gamma (float): Focal Loss 的聚焦系数，默认 2.0。越大越关注难样本。
        smoothing (float): 标签平滑系数，默认 0.1。范围为 [0, 1]。
        ignore_index (int64): 指定忽略的目标值，默认 255。
        edge_label (bool): 是否使用边缘标签（保留兼容性）。
    """

    def __init__(self, gamma=2.0, smoothing=0.1, ignore_index=255, edge_label=False):
        super(FocalLoss_Smooth, self).__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.edge_label = edge_label

    def forward(self, logit, label):
        # 1. 重塑 logit 和 label，保持与你原有代码一致的处理流程
        logit = paddle.reshape(
            logit, [logit.shape[0], logit.shape[1], -1])  # N,C,H,W => N,C,H*W
        logit = paddle.transpose(logit, [0, 2, 1])  # N,C,H*W => N,H*W,C
        logit = paddle.reshape(logit,
                               [-1, logit.shape[2]])  # N,H*W,C => N*H*W,C
        
        label = paddle.reshape(label, [-1, 1])
        label = paddle.cast(label, dtype='int64')
        
        # 2. 处理忽略索引
        if self.ignore_index is not None:
            valid_mask = (label != self.ignore_index).astype('float32')
            label_clone = label.clone()
            label_clone[label == self.ignore_index] = 0  # 临时替换避免 one-hot 出错
        else:
            valid_mask = paddle.ones_like(label).astype('float32')
            label_clone = label
        
        num_classes = logit.shape[1]
        
        # 3. 生成平滑标签
        # 原始 one-hot
        label_onehot = F.one_hot(label_clone.reshape([-1]), num_classes)
        label_onehot = paddle.cast(label_onehot, dtype='float32')
        
        # 应用 Label Smoothing: target = (1 - smoothing) * one_hot + smoothing / num_classes
        smooth_label = (1 - self.smoothing) * label_onehot + self.smoothing / num_classes
        
        # 恢复忽略像素的标签为全零（不计入损失）
        if self.ignore_index is not None:
            smooth_label = smooth_label * valid_mask.reshape([-1, 1])
        
        # 4. 计算 Log Softmax
        log_prob = F.log_softmax(logit, axis=-1)
        
        # 5. 计算 Focal Loss 的核心组件
        prob = paddle.exp(log_prob)  # softmax 概率
        prob = paddle.clip(prob, min=1e-7, max=1.0)  # 数值稳定性
        
        # 计算每个样本在其平滑标签下的预测概率 pt
        # pt = sum(target_smooth * prob)，即真实类别（平滑后）对应的概率
        pt = paddle.sum(smooth_label * prob, axis=-1)
        
        # Focal Weight: (1 - pt)^gamma
        focal_weight = paddle.pow(1 - pt, self.gamma)
        
        # 6. 计算 Cross Entropy: -sum(target_smooth * log_prob)
        cross_entropy = -paddle.sum(smooth_label * log_prob, axis=-1)
        
        # 7. 组合 Focal Loss
        loss = focal_weight * cross_entropy
        
        # 8. 应用有效像素 mask 并计算均值
        if self.ignore_index is not None:
            valid_mask_flat = valid_mask.reshape([-1])
            loss = loss * valid_mask_flat
            avg_loss = paddle.sum(loss) / (paddle.sum(valid_mask_flat) + 1e-8)
        else:
            avg_loss = paddle.mean(loss)
        
        return avg_loss