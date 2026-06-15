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
class Poly1Loss_Smooth(nn.Layer):
    """Poly1 Loss = CE + epsilon * (1 - pt)"""
    def __init__(self, epsilon=1.0, smoothing=0.1, ignore_index=255):
        super().__init__()
        self.epsilon = epsilon
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, logit, label):
        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)
        
        n, c, h, w = logit.shape
        label = label.reshape((-1,))
        valid_mask = (label != self.ignore_index)
        
        label_clean = label * valid_mask.astype('int64')
        target = F.one_hot(label_clean, c).astype('float32')
        target = (1 - self.smoothing) * target + self.smoothing / c
        
        logit_flat = logit.transpose((0, 2, 3, 1)).reshape((-1, c))
        prob = F.softmax(logit_flat, axis=-1)
        log_prob = F.log_softmax(logit_flat, axis=-1)
        
        pt = paddle.sum(target * prob, axis=-1)
        ce = -paddle.sum(target * log_prob, axis=-1)
        poly_term = self.epsilon * (1 - pt)  # 一阶 Poly
        
        loss = (ce + poly_term) * valid_mask.astype('float32')
        return paddle.sum(loss) / (paddle.sum(valid_mask) + 1e-5)