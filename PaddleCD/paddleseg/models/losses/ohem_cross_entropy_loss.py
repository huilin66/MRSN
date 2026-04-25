# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager

@manager.LOSSES.add_component
class OhemCrossEntropyLoss_Smooth(nn.Layer):
    """
    结合了标签平滑（Label Smoothing）和在线难样本挖掘（OHEM）的交叉熵损失函数。
    
    Args:
        smoothing (float, optional): 标签平滑系数. 默认: 0.1。
        thresh (float, optional): OHEM 的阈值，概率低于此值的被视为难样本. 默认: 0.7。
        min_kept (int, optional): 每次计算 Loss 至少保留的像素数量. 默认: 10000。
        ignore_index (int64, optional): 忽略的类别索引. 默认: 255。
    """

    def __init__(self, smoothing=0.1, thresh=0.7, min_kept=10000, ignore_index=255):
        super(OhemCrossEntropyLoss_Smooth, self).__init__()
        self.smoothing = smoothing
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.EPS = 1e-5

    def forward(self, logit, label):
        """
        前向计算。
        logit shape: (N, C, H, W)
        label shape: (N, H, W)
        """
        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)

        n, c, h, w = logit.shape
        label = label.reshape((-1, ))
        
        # 1. 基础 Mask，数据类型设为 int64 供逻辑运算使用
        valid_mask_int = (label != self.ignore_index).astype('int64')
        num_valid = valid_mask_int.sum()

        # 这里 int64 * int64 是允许的
        label_for_mining = label * valid_mask_int 
        
        # --- 挖掘逻辑 ---
        prob = F.softmax(logit, axis=1)
        prob = prob.transpose((1, 0, 2, 3)).reshape((c, -1))

        # 在与 prob (float32) 相加时，将 mask 转为 float32
        valid_mask_float = valid_mask_int.astype('float32')
        prob_with_ignore = prob + (1.0 - valid_mask_float)
        
        # 获取对应标签位置的预测概率
        label_onehot = F.one_hot(label_for_mining, c).transpose((1, 0))
        target_prob = paddle.sum(prob_with_ignore * label_onehot, axis=0)

        threshold = self.thresh
        if self.min_kept < num_valid and num_valid > 0:
            if self.min_kept > 0:
                index = target_prob.argsort()
                # 取出阈值索引张量
                threshold_index_tensor = index[min(len(index), self.min_kept) - 1]
                # 修复 IndexError: 0维张量不能用 [0]，应使用 .item() 转换为 Python 原生 int
                threshold_index = int(threshold_index_tensor.item()) 
                
                if target_prob[threshold_index] > self.thresh:
                    threshold = target_prob[threshold_index]
            
            # 只有概率小于 threshold 的像素被保留 (kept_mask)
            kept_mask = (target_prob < threshold).astype('int64')
            # 修复 NameError: valid_mask 改为 valid_mask_int
            final_mask_int = valid_mask_int * kept_mask
        else:
            final_mask_int = valid_mask_int

        # 2. 计算 Label Smoothing Loss (平滑逻辑)
        # 将 logit 展平为 (N*H*W, C)
        logit_flat = logit.transpose((0, 2, 3, 1)).reshape((-1, c))
        
        # 生成平滑后的 Soft Label
        target_smooth = F.one_hot(label_for_mining, c)
        target_smooth = paddle.cast(target_smooth, 'float32')
        target_smooth = (1 - self.smoothing) * target_smooth + self.smoothing / c
        
        # 计算 Log Softmax
        log_prob = F.log_softmax(logit_flat, axis=-1)
        
        # 计算每个像素的 Cross Entropy: -sum(target_smooth * log_prob)
        loss = -paddle.sum(target_smooth * log_prob, axis=-1)
        
        # --- 最终 Mask 处理与计算 ---
        # 修复 TypeError: 将用于乘法计算的 final_mask 转换为 float32
        final_mask_float = final_mask_int.astype('float32')
        
        # 应用最终的 OHEM Mask
        loss = loss * final_mask_float
        
        # 平均损失
        avg_loss = paddle.sum(loss) / (paddle.sum(final_mask_float) + self.EPS)

        label.stop_gradient = True
        return avg_loss


@manager.LOSSES.add_component
class OhemCrossEntropyLoss(nn.Layer):
    """
    Implements the ohem cross entropy loss function.

    Args:
        thresh (float, optional): The threshold of ohem. Default: 0.7.
        min_kept (int, optional): The min number to keep in loss computation. Default: 10000.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, thresh=0.7, min_kept=10000, ignore_index=255):
        super(OhemCrossEntropyLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.EPS = 1e-5

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)

        # get the label after ohem
        n, c, h, w = logit.shape
        label = label.reshape((-1, ))
        valid_mask = (label != self.ignore_index).astype('int64')
        num_valid = valid_mask.sum()
        label = label * valid_mask

        prob = F.softmax(logit, axis=1)
        prob = prob.transpose((1, 0, 2, 3)).reshape((c, -1))

        if self.min_kept < num_valid and num_valid > 0:
            # let the value which ignored greater than 1
            prob = prob + (1 - valid_mask)

            # get the prob of relevant label
            label_onehot = F.one_hot(label, c)
            label_onehot = label_onehot.transpose((1, 0))
            prob = prob * label_onehot
            prob = paddle.sum(prob, axis=0)

            threshold = self.thresh
            if self.min_kept > 0:
                index = prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                threshold_index = int(threshold_index.numpy()[0])
                if prob[threshold_index] > self.thresh:
                    threshold = prob[threshold_index]
                kept_mask = (prob < threshold).astype('int64')
                label = label * kept_mask
                valid_mask = valid_mask * kept_mask

        # make the invalid region as ignore
        label = label + (1 - valid_mask) * self.ignore_index

        label = label.reshape((n, 1, h, w))
        valid_mask = valid_mask.reshape((n, 1, h, w)).astype('float32')
        loss = F.softmax_with_cross_entropy(
            logit, label, ignore_index=self.ignore_index, axis=1)
        loss = loss * valid_mask
        avg_loss = paddle.mean(loss) / (paddle.mean(valid_mask) + self.EPS)

        label.stop_gradient = True
        valid_mask.stop_gradient = True
        return avg_loss
