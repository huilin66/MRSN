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
class DiceLoss(nn.Layer):
    """
    Implements the dice loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        smooth (float32): laplace smoothing,
            to smooth dice loss and accelerate convergence. following:
            https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
    """

    def __init__(self, ignore_index=255, smooth=0.):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5
        self.smooth = smooth

    def forward(self, logits, labels):
        labels = paddle.cast(labels, dtype='int32')
        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1])
        labels_one_hot = paddle.transpose(labels_one_hot, [0, 3, 1, 2])
        labels_one_hot = paddle.cast(labels_one_hot, dtype='float32')

        logits = F.softmax(logits, axis=1)

        mask = (paddle.unsqueeze(labels, 1) != self.ignore_index)
        mask = mask.astype('float32')
        logits = logits * mask
        labels_one_hot = labels_one_hot * mask

        dims = (0, ) + tuple(range(2, labels.ndimension() + 1))

        intersection = paddle.sum(logits * labels_one_hot, dims)
        cardinality = paddle.sum(logits + labels_one_hot, dims)
        dice_loss = ((2. * intersection + self.smooth) /
                     (cardinality + self.eps + self.smooth)).mean()
        return 1 - dice_loss


@manager.LOSSES.add_component
class HardDiceLoss(nn.Layer):
    """
    困难样本感知的 Dice Loss，通过加权或平方增强困难样本的梯度。
    
    相比标准 Dice Loss，本实现提供三种增强模式：
    - 'focal': Focal-style 加权，对低 dice 的类别/区域给予更高权重
    - 'square': 使用平方项放大困难样本的梯度信号
    - 'both': 同时使用 focal 加权和平方项

    Args:
        ignore_index (int64): 指定忽略的目标值. 默认 ``255``.
        smooth (float32): Laplace 平滑系数，加速收敛. 默认 ``1.0``.
        mode (str): 增强模式，可选 'focal', 'square', 'both'. 默认 ``'focal'``.
        focal_gamma (float): mode 为 'focal' 或 'both' 时的聚焦系数. 默认 ``1.0``.
        apply_softmax (bool): 是否在 loss 内部做 softmax. 默认 ``True``.
    """

    def __init__(self, 
                 ignore_index=255, 
                 smooth=1.0,
                 mode='focal',
                 focal_gamma=1.0,
                 apply_softmax=True):
        super(HardDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5
        self.smooth = smooth
        self.mode = mode
        self.focal_gamma = focal_gamma
        self.apply_softmax = apply_softmax
        
        assert mode in ['focal', 'square', 'both'], \
            f"mode must be 'focal', 'square' or 'both', but got {mode}"

    def forward(self, logits, labels):
        # 1. 预处理 labels 为 one-hot 格式
        labels = paddle.cast(labels, dtype='int32')
        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1])
        labels_one_hot = paddle.transpose(labels_one_hot, [0, 3, 1, 2])
        labels_one_hot = paddle.cast(labels_one_hot, dtype='float32')

        # 2. 应用 softmax（可选，兼容外部已做 softmax 的情况）
        if self.apply_softmax:
            logits = F.softmax(logits, axis=1)

        # 3. 处理 ignore_index mask
        if self.ignore_index is not None:
            mask = (paddle.unsqueeze(labels, 1) != self.ignore_index)
            mask = mask.astype('float32')
            logits = logits * mask
            labels_one_hot = labels_one_hot * mask
        else:
            mask = paddle.ones_like(labels_one_hot)

        # 4. 计算每个类别的 Dice 分量
        dims = (0, ) + tuple(range(2, labels.ndimension() + 1))
        
        intersection = paddle.sum(logits * labels_one_hot, dims)
        cardinality = paddle.sum(logits + labels_one_hot, dims)
        
        # 基础 Dice 分数（每个类别独立计算）
        dice_per_class = (2. * intersection + self.smooth) / \
                         (cardinality + self.eps + self.smooth)
        
        # 5. 根据 mode 增强困难样本的梯度
        if self.mode == 'focal':
            # Focal-style: 对 dice 低的类别给予更高权重
            # dice_loss_per_class = 1 - dice_per_class （每个类别的损失）
            # 困难类别 dice 小 -> loss 大 -> focal weight 大
            focal_weight = paddle.pow(1.0 - dice_per_class.detach(), self.focal_gamma)
            dice_loss_per_class = 1.0 - dice_per_class
            weighted_loss = focal_weight * dice_loss_per_class
            loss = paddle.mean(weighted_loss)
            
        elif self.mode == 'square':
            # 平方模式: 直接用平方增大困难区域的梯度
            # 对 intersection 和 cardinality 做平方
            intersection_sq = paddle.sum(logits * logits * labels_one_hot, dims)
            cardinality_sq = paddle.sum(logits * logits + labels_one_hot, dims)
            
            dice_sq = (2. * intersection_sq + self.smooth) / \
                      (cardinality_sq + self.eps + self.smooth)
            loss = 1.0 - paddle.mean(dice_sq)
            
        elif self.mode == 'both':
            # 同时使用 focal 和 square
            focal_weight = paddle.pow(1.0 - dice_per_class.detach(), self.focal_gamma)
            
            intersection_sq = paddle.sum(logits * logits * labels_one_hot, dims)
            cardinality_sq = paddle.sum(logits * logits + labels_one_hot, dims)
            
            dice_sq = (2. * intersection_sq + self.smooth) / \
                      (cardinality_sq + self.eps + self.smooth)
            
            dice_loss_per_class = 1.0 - dice_sq
            weighted_loss = focal_weight * dice_loss_per_class
            loss = paddle.mean(weighted_loss)

        return loss