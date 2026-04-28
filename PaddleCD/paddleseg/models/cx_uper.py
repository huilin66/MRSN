import paddle
from paddle import nn, Tensor
from paddle.nn import functional as F
from typing import Tuple


from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.cvlibs.param_init import KaimingInitMixin
from paddleseg.models.layers.blocks import Conv3x3, Conv1x1, get_norm_layer, Identity, make_norm
from paddleseg.models.layers.attention import CBAM
from paddleseg.models.backbones import resnet, convnext


@manager.MODELS.add_component
class CX_Uper_4B(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    hsi_chs=242, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper_4B, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone0 = convnext.convnext_tiny()
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny(in_chans=2)
            self.backbone3 = convnext.convnext_tiny(in_chans=hsi_chs)
        elif backb == 'convnext_small':
            self.backbone0 = convnext.convnext_small()
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small(in_chans=2)
            self.backbone3 = convnext.convnext_small(in_chans=hsi_chs)
        else:
            self.backbone0 = convnext.convnext_base()
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base(in_chans=2)
            self.backbone3 = convnext.convnext_base(in_chans=hsi_chs)
        

        self.decode_head2b = UPerHead_3B(self.backbone1.dims[:3], num_classes=num_classes)
        self.decode_head3b = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        t3 = t2
        t0 = paddle.concat([t1[:, :2, ...], t1[:, 3:4, ...]], axis=1)
        t2 = t1[:, 4:, ...]
        t1 = t1[:, :3, ...]
        fs0 = self.backbone0(t0)
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f0, f1, f2 in zip(fs0, fs1, fs2):
            f = self.drop(paddle.concat([f0, f1, f2], axis=1))
            fs_diff.append(f)
        y2 = self.decode_head2b(fs_diff)
        y3 = self.decode_head3b(fs3)
        y = y2+y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper_3B_1(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 backb,
                 hsi_chs=242,
                 dropout_rate=0.0,
                 ):
        super(CX_Uper_3B_1, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny(in_chans=2)
            self.backbone3 = convnext.convnext_tiny(in_chans=hsi_chs)
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small(in_chans=2)
            self.backbone3 = convnext.convnext_small(in_chans=hsi_chs)
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base(in_chans=2)
            self.backbone3 = convnext.convnext_base(in_chans=hsi_chs)

        self.decode_head2b = UPerHead_3B(self.backbone1.dims[:3], num_classes=num_classes)
        self.decode_head3b = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        t3 = t2
        t0 = paddle.concat([t1[:, :2, ...], t1[:, 3:4, ...]], axis=1)
        t2 = t1[:, 4:, ...]
        t1 = t1[:, :3, ...]
        fs0 = self.backbone1(t0)
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f0, f1, f2 in zip(fs0, fs1, fs2):
            f = self.drop(paddle.concat([f0, f1, f2], axis=1))
            fs_diff.append(f)
        y2 = self.decode_head2b(fs_diff)
        y3 = self.decode_head3b(fs3)
        y = y2+y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper_3B_2(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 backb,
                 hsi_chs=242,
                 dropout_rate=0.0,
                 ):
        super(CX_Uper_3B_2, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny(in_chans=4)
            self.backbone2 = convnext.convnext_tiny(in_chans=2)
            self.backbone3 = convnext.convnext_tiny(in_chans=hsi_chs)
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small(in_chans=4)
            self.backbone2 = convnext.convnext_small(in_chans=2)
            self.backbone3 = convnext.convnext_small(in_chans=hsi_chs)
        else:
            self.backbone1 = convnext.convnext_base(in_chans=4)
            self.backbone2 = convnext.convnext_base(in_chans=2)
            self.backbone3 = convnext.convnext_base(in_chans=hsi_chs)

        self.decode_head2b = UPerHead_2B(self.backbone1.dims[:3], num_classes=num_classes)
        self.decode_head3b = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        t3 = t2
        t2 = t1[:, 4:, ...]
        t1 = t1[:, :4, ...]
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            f = self.drop(paddle.concat([f1, f2], axis=1))
            fs_diff.append(f)
        y2 = self.decode_head2b(fs_diff)
        y3 = self.decode_head3b(fs3)
        y = y2+y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]

@manager.MODELS.add_component
class CX_Uper_4B_CA(CX_Uper_4B):
    def __init__(self, **kwargs):
        super(CX_Uper_4B_CA, self).__init__(**kwargs)

        # ⚠️ 修复 1：在 Paddle/PyTorch 中，保存子网络层不能用普通列表 []
        # 必须用 nn.LayerList()，否则这些 Attention 层的参数不会被优化器更新！
        self.cas1 = nn.LayerList()
        self.cas2 = nn.LayerList()

        for dim in self.backbone1.dims[:3]:
            ca1 = nn.MultiHeadAttention(dim * 3, 8, kdim=dim, vdim=dim)
            ca2 = nn.MultiHeadAttention(dim, 8, kdim=dim * 3, vdim=dim * 3)
            self.cas1.append(ca1)
            self.cas2.append(ca2)

    def forward(self, t1, t2):
        t3 = t2
        t0 = paddle.concat([t1[:, :2, ...], t1[:, 3:4, ...]], axis=1)
        t2 = t1[:, 4:, ...]
        t1 = t1[:, :3, ...]

        fs0 = self.backbone0(t0)
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)

        fs_diff = []
        for f0, f1, f2 in zip(fs0, fs1, fs2):
            f = self.drop(paddle.concat([f0, f1, f2], axis=1))
            fs_diff.append(f)

        fs_diff_ca = []
        fs3_ca = []

        for ca1, ca2, f1, f2 in zip(self.cas1, self.cas2, fs_diff, fs3):
            shape1, shape2 = f1.shape, f2.shape

            # 🚀 优化：使用自适应池化来缩小 Key 和 Value 的空间分辨率
            # 将 (H, W) 压缩到 (32, 32)，这样序列长度从 H*W 骤降到 1024！
            # Query 的形状保持不变，确保输出的分辨率不缩水。
            pool_size = (32, 32)  # 如果还是爆显存，可以改成 (16, 16)
            f1_kv_pool = F.adaptive_avg_pool2d(f1, pool_size)
            f2_kv_pool = F.adaptive_avg_pool2d(f2, pool_size)

            # --- 处理 Q (保持原始大分辨率) ---
            f1_q = f1.reshape((shape1[0], shape1[1], -1)).transpose((0, 2, 1))  # [B, H*W, C1]
            f2_q = f2.reshape((shape2[0], shape2[1], -1)).transpose((0, 2, 1))  # [B, H*W, C2]

            # --- 处理 K 和 V (使用池化后的小分辨率) ---
            f1_kv = f1_kv_pool.reshape((shape1[0], shape1[1], -1)).transpose((0, 2, 1))  # [B, 1024, C1]
            f2_kv = f2_kv_pool.reshape((shape2[0], shape2[1], -1)).transpose((0, 2, 1))  # [B, 1024, C2]

            # --- 计算 Attention ---
            # ca1: f1 作为 Query, f2 作为 Key 和 Value
            ff1 = ca1(f1_q, f2_kv, f2_kv)
            ff1 = ff1.transpose((0, 2, 1)).reshape((shape1[0], shape1[1], shape1[2], shape1[3]))

            # ca2: f2 作为 Query, f1 作为 Key 和 Value
            ff2 = ca2(f2_q, f1_kv, f1_kv)
            ff2 = ff2.transpose((0, 2, 1)).reshape((shape2[0], shape2[1], shape2[2], shape2[3]))

            fs_diff_ca.append(ff1)
            fs3_ca.append(ff2)

        # ⚠️ 修复 2：逻辑错误
        # 原代码写的是 self.decode_head2b(fs_diff)，导致 Attention 算完后被抛弃了！
        # 必须把计算出的 fs_diff_ca 和 fs3_ca 传给解码头
        y2 = self.decode_head2b(fs_diff_ca)
        y3 = self.decode_head3b(fs3_ca)

        y = y2 + y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


class FlashMultiHeadAttention(nn.Layer):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, causal=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.dropout = dropout

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.head_dim = embed_dim // num_heads  # 自动计算每个头的维度

        # QKV 线性投影（和标准 MHA 完全一样）
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, q, k=None, v=None):
        if k is None:
            k = q
        if v is None:
            v = q

        B, S_q, _ = q.shape  # [batch, seq_len, embed_dim]
        B, S_k, _ = k.shape

        q = self.q_proj(q)  # [B, S, D]
        k = self.k_proj(k)
        v = self.v_proj(v)


        q = q.reshape((B, -1, self.num_heads, self.head_dim))
        k = k.reshape((B, -1, self.num_heads, self.head_dim))
        v = v.reshape((B, -1, self.num_heads, self.head_dim))

        # 3. 调用 FlashAttention 核心 API
        # 该函数在底层会自动选择最优 Kernel，不产生中间的 N*N 矩阵
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            training=self.training
        )
        attn_out_reshape = attn_out.reshape([B, S_q, self.embed_dim])

        # 5. 输出投影
        output = self.out_proj(attn_out_reshape)
        return output

@manager.MODELS.add_component
class CX_Uper_4B_CA_FLASH(CX_Uper_4B):
    def __init__(self, **kwargs):
        super(CX_Uper_4B_CA_FLASH, self).__init__(**kwargs)

        self.cas1 = []
        self.cas2 = []
        for dim in self.backbone1.dims[:3]:
            ca1 = FlashMultiHeadAttention(dim*3, 8, kdim=dim, vdim=dim)
            ca2 = FlashMultiHeadAttention(dim, 8, kdim=dim*3, vdim=dim*3)
            self.cas1.append(ca1)
            self.cas2.append(ca2)

    def forward(self, t1, t2):
        t3 = t2
        t0 = paddle.concat([t1[:, :2, ...], t1[:, 3:4, ...]], axis=1)
        t2 = t1[:, 4:, ...]
        t1 = t1[:, :3, ...]
        fs0 = self.backbone0(t0)
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f0, f1, f2 in zip(fs0, fs1, fs2):
            f = self.drop(paddle.concat([f0, f1, f2], axis=1))
            fs_diff.append(f)

        fs_diff_ca = []
        fs3_ca = []
        for ca1, ca2, f1, f2 in zip(self.cas1, self.cas2, fs_diff, fs3):
            shape1, shape2 = f1.shape, f2.shape
            # print(f1.shape, f2.shape)
            f1 = f1.reshape((shape1[0], shape1[1], -1)).transpose((0, 2, 1))
            f2 = f2.reshape((shape2[0], shape2[1], -1)).transpose((0, 2, 1))
            # print(f1.shape, f2.shape)
            # print(ca1, ca2)
            ff1 = ca1(f1, f2, f2)
            ff1 = ff1.transpose((0, 2, 1)).reshape((shape1[0], shape1[1], shape1[2], shape1[3]))
            ff2 = ca2(f2, f1, f1)
            ff2 = ff2.transpose((0, 2, 1)).reshape((shape2[0], shape2[1], shape2[2], shape2[3]))
            # print(ff1.shape, ff2.shape)
            fs_diff_ca.append(ff1)
            fs3_ca.append(ff2)
        y2 = self.decode_head2b(fs_diff)
        y3 = self.decode_head3b(fs3)
        y = y2+y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]

@manager.MODELS.add_component
class CX_Uper_4B_CA_FULL(CX_Uper_4B):
    def __init__(self, **kwargs):
        super(CX_Uper_4B_CA_FULL, self).__init__(**kwargs)

        self.cas1 = []
        self.cas2 = []
        for dim in self.backbone1.dims[:3]:
            ca1 = nn.MultiHeadAttention(dim*3, 8, kdim=dim, vdim=dim)
            ca2 = nn.MultiHeadAttention(dim, 8, kdim=dim*3, vdim=dim*3)
            self.cas1.append(ca1)
            self.cas2.append(ca2)

    def forward(self, t1, t2):
        t3 = t2
        t0 = paddle.concat([t1[:, :2, ...], t1[:, 3:4, ...]], axis=1)
        t2 = t1[:, 4:, ...]
        t1 = t1[:, :3, ...]
        fs0 = self.backbone0(t0)
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f0, f1, f2 in zip(fs0, fs1, fs2):
            f = self.drop(paddle.concat([f0, f1, f2], axis=1))
            fs_diff.append(f)

        fs_diff_ca = []
        fs3_ca = []
        for ca1, ca2, f1, f2 in zip(self.cas1, self.cas2, fs_diff, fs3):
            shape1, shape2 = f1.shape, f2.shape
            # print(f1.shape, f2.shape)
            f1 = f1.reshape((shape1[0], shape1[1], -1)).transpose((0, 2, 1))
            f2 = f2.reshape((shape2[0], shape2[1], -1)).transpose((0, 2, 1))
            # print(f1.shape, f2.shape)
            # print(ca1, ca2)
            ff1 = ca1(f1, f2, f2)
            ff1 = ff1.transpose((0, 2, 1)).reshape((shape1[0], shape1[1], shape1[2], shape1[3]))
            ff2 = ca2(f2, f1, f1)
            ff2 = ff2.transpose((0, 2, 1)).reshape((shape2[0], shape2[1], shape2[2], shape2[3]))
            # print(ff1.shape, ff2.shape)
            fs_diff_ca.append(ff1)
            fs3_ca.append(ff2)
        y2 = self.decode_head2b(fs_diff)
        y3 = self.decode_head3b(fs3)
        y = y2+y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper_4B_CA_SRA(CX_Uper_4B):
    def __init__(self, **kwargs):
        super(CX_Uper_4B_CA_SRA, self).__init__(**kwargs)

        self.cas1 = nn.LayerList()
        self.cas2 = nn.LayerList()

        # 添加用于降维的卷积层 (保持空间相对位置)
        self.sr1 = nn.LayerList()
        self.sr2 = nn.LayerList()

        for dim in self.backbone1.dims[:3]:
            ca1 = nn.MultiHeadAttention(dim * 3, 8, kdim=dim, vdim=dim)
            ca2 = nn.MultiHeadAttention(dim, 8, kdim=dim * 3, vdim=dim * 3)
            self.cas1.append(ca1)
            self.cas2.append(ca2)

            # Stride=2 缩小一半，如果是极度缺显存可以设为 Stride=4
            self.sr1.append(nn.Conv2D(dim, dim, kernel_size=3, stride=2, padding=1))
            self.sr2.append(nn.Conv2D(dim * 3, dim * 3, kernel_size=3, stride=2, padding=1))

    def forward(self, t1, t2):
        t3 = t2
        t0 = paddle.concat([t1[:, :2, ...], t1[:, 3:4, ...]], axis=1)
        t2 = t1[:, 4:, ...]
        t1 = t1[:, :3, ...]
        fs0 = self.backbone0(t0)
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f0, f1, f2 in zip(fs0, fs1, fs2):
            f = self.drop(paddle.concat([f0, f1, f2], axis=1))
            fs_diff.append(f)

        fs_diff_ca = []
        fs3_ca = []

        for ca1, ca2, sr1, sr2, f1, f2 in zip(self.cas1, self.cas2, self.sr1, self.sr2, fs_diff, fs3):
            shape1, shape2 = f1.shape, f2.shape

            # --- Q 保持全分辨率不变 ---
            f1_q = f1.reshape((shape1[0], shape1[1], -1)).transpose((0, 2, 1))
            f2_q = f2.reshape((shape2[0], shape2[1], -1)).transpose((0, 2, 1))

            # --- K和V 使用卷积进行平滑降维 ---
            f1_kv = sr2(f1)  # 注意通道数匹配
            f2_kv = sr1(f2)

            f1_kv = f1_kv.reshape((f1_kv.shape[0], f1_kv.shape[1], -1)).transpose((0, 2, 1))
            f2_kv = f2_kv.reshape((f2_kv.shape[0], f2_kv.shape[1], -1)).transpose((0, 2, 1))

            # --- 计算 Attention ---
            ff1 = ca1(f1_q, f2_kv, f2_kv)
            ff1 = ff1.transpose((0, 2, 1)).reshape((shape1[0], shape1[1], shape1[2], shape1[3]))

            ff2 = ca2(f2_q, f1_kv, f1_kv)
            ff2 = ff2.transpose((0, 2, 1)).reshape((shape2[0], shape2[1], shape2[2], shape2[3]))

            fs_diff_ca.append(ff1)
            fs3_ca.append(ff2)

        # 传入 decode_head
        y2 = self.decode_head2b(fs_diff_ca)
        y3 = self.decode_head3b(fs3_ca)
        y = y2+y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]

# @manager.MODELS.add_component
# class CX_Uper_4B_CA_Flash(CX_Uper_4B):
#     def __init__(self, **kwargs):
#         super(CX_Uper_4B_CA_Flash, self).__init__(**kwargs)
#
#         self.num_heads = 8
#         self.q_projs = nn.LayerList()
#         self.k_projs = nn.LayerList()
#         self.v_projs = nn.LayerList()
#         self.out_projs = nn.LayerList()
#
#         # 为 backbone 的前三个 stage 定义 Cross-Attention 映射
#         # 这里以 ca1 (f1 为 Query, f2 为 Key/Value) 为例演示
#         # 如果需要双向 Cross-Attention，可以按照相同逻辑增加 ca2 的映射层
#         for dim in self.backbone1.dims[:3]:
#             # 输入维度：f1 为 dim*3 (concat后的), f2 为 dim
#             self.q_projs.append(nn.Linear(dim * 3, dim * 3))
#             self.k_projs.append(nn.Linear(dim, dim * 3))
#             self.v_projs.append(nn.Linear(dim, dim * 3))
#             self.out_projs.append(nn.Linear(dim * 3, dim * 3))
#
#     def _flash_attn_block(self, q_feat, kv_feat, q_proj, k_proj, v_proj, out_proj):
#         """
#         核心 FlashAttention 实现
#         q_feat: [B, C_q, H, W]
#         kv_feat: [B, C_kv, H, W]
#         """
#         B, C, H, W = q_feat.shape
#         N = H * W
#
#         # 1. 展平并映射
#         # [B, C, H, W] -> [B, N, C]
#         q = q_feat.reshape((B, C, N)).transpose((0, 2, 1))
#         kv = kv_feat.reshape((kv_feat.shape[0], kv_feat.shape[1], -1)).transpose((0, 2, 1))
#
#         q = q_proj(q)
#         k = k_proj(kv)
#         v = v_proj(kv)
#
#         # 2. 拆分多头 [B, N, num_heads, head_dim]
#         head_dim = C // self.num_heads
#         q = q.reshape((B, N, self.num_heads, head_dim)).transpose((0, 2, 1, 3))
#         k = k.reshape((B, k.shape[1], self.num_heads, head_dim)).transpose((0, 2, 1, 3))
#         v = v.reshape((B, v.shape[1], self.num_heads, head_dim)).transpose((0, 2, 1, 3))
#
#         # 3. 调用 FlashAttention 核心 API
#         # 该函数在底层会自动选择最优 Kernel，不产生中间的 N*N 矩阵
#         attn_out = F.scaled_dot_product_attention(
#             q, k, v,
#             attn_mask=None,
#             dropout_p=0.0,
#             is_causal=False,
#             training=self.training
#         )
#
#         # 4. 合并头并还原形状
#         attn_out = attn_out.transpose((0, 2, 1, 3)).reshape((B, N, C))
#         attn_out = out_proj(attn_out)
#         attn_out = attn_out.transpose((0, 2, 1)).reshape((B, C, H, W))
#
#         return attn_out
#
#     def forward(self, t1, t2):
#         t3 = t2
#         t0 = paddle.concat([t1[:, :2, ...], t1[:, 3:4, ...]], axis=1)
#         t2 = t1[:, 4:, ...]
#         t1 = t1[:, :3, ...]
#
#         # 骨干特征提取
#         fs0 = self.backbone0(t0)
#         fs1 = self.backbone1(t1)
#         fs2 = self.backbone2(t2)
#         fs3 = self.backbone3(t3)
#
#         fs_diff = []
#         for f0, f1, f2 in zip(fs0, fs1, fs2):
#             f = self.drop(paddle.concat([f0, f1, f2], axis=1))
#             fs_diff.append(f)
#
#         # 执行 Flash Cross-Attention
#         fs_diff_ca = []
#         for i in range(len(fs_diff)):
#             # 这里实现 fs_diff 对 fs3 的交叉注意力
#             ff1 = self._flash_attn_block(
#                 fs_diff[i], fs3[i],
#                 self.q_projs[i], self.k_projs[i], self.v_projs[i], self.out_projs[i]
#             )
#             fs_diff_ca.append(ff1)
#
#         # 解码头
#         y2 = self.decode_head2b(fs_diff_ca)
#         y3 = self.decode_head3b(fs3)  # 若 fs3 也需要 CA，同理增加映射层即可
#
#         y = y2 + y3
#         out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
#
#         return [out]

@manager.MODELS.add_component
class CX_Uper_2B(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper_2B, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head = UPerHead_2B(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            fs_diff.append(self.drop(paddle.concat([f1, f2], axis=1)))
        y = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]

@manager.MODELS.add_component
class CX_Uper_2B_cat(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper_2B_cat, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head = UPerHead_2B(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            fs_diff.append(self.drop(paddle.concat([f1, f2], axis=1)))
        y = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]

@manager.MODELS.add_component
class CX_Uper_2B_plus(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper_2B_plus, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head1 = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.decode_head2 = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        y1 = self.decode_head1(fs1)
        y2 = self.decode_head2(fs1)
        y = y1+y2
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]


@manager.MODELS.add_component
class CX_Uper_2B_ca1(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper_2B_ca1, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

        self.cas = []
        for dim in self.backbone1.dims[:3]:
            ca = nn.MultiHeadAttention(dim, 4)
            self.cas.append(ca)


    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        outs = []
        for ca, f1, f2 in zip(self.cas, fs1, fs2):
            shp = f1.shape
            f1 = f1.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            f2 = f2.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            f1 = ca(f1, f2, f1)
            f1 = f1.transpose((0, 2, 1)).reshape((shp[0], shp[1], shp[2], shp[3]))
            outs.append(f1)
        y = self.decode_head(outs)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]

class CX_Uper_2B_ca2(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper_2B_ca2, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

        self.cas = []
        for dim in self.backbone1.dims[:3]:
            ca = nn.MultiHeadAttention(dim, 4)
            self.cas.append(ca)


    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        outs = []
        for ca, f1, f2 in zip(self.cas, fs1, fs2):
            shp = f1.shape
            f1 = f1.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            f2 = f2.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            f2 = ca(f2, f1, f2)
            f2 = f2.transpose((0, 2, 1)).reshape((shp[0], shp[1], shp[2], shp[3]))
            outs.append(f2)
        y = self.decode_head(outs)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]

@manager.MODELS.add_component
class CX_Uper_2B_ca_cat(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper_2B_ca_cat, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head = UPerHead_2B(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

        self.cas = []
        for dim in self.backbone1.dims[:3]:
            ca = nn.MultiHeadAttention(dim, 4)
            self.cas.append(ca)


    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        outs = []
        for ca, f1, f2 in zip(self.cas, fs1, fs2):
            shp = f1.shape
            f1 = f1.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            f2 = f2.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            ff2 = ca(f2, f1, f2)
            ff2 = ff2.transpose((0, 2, 1)).reshape((shp[0], shp[1], shp[2], shp[3]))
            ff1 = ca(f1, f2, f1)
            ff1 = ff1.transpose((0, 2, 1)).reshape((shp[0], shp[1], shp[2], shp[3]))
            outs.append(self.drop(paddle.concat([ff1, ff2], axis=1)))
        y = self.decode_head(outs)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]

@manager.MODELS.add_component
class CX_Uper_2B1(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper_2B1, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
        
        self.decode_head = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        y = self.decode_head(fs1)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]

@manager.MODELS.add_component
class CX_Uper_3B_cat(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper_3B_cat, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
            self.backbone3 = convnext.convnext_tiny(in_chans=242)
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
            self.backbone3 = convnext.convnext_small(in_chans=242)
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
            self.backbone3 = convnext.convnext_base(in_chans=242)
        
        self.decode_head = UPerHead_3B(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        t3 = t2
        t2 = t1[:, 3:, ...]
        t1 = t1[:, :3, ...]
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f1, f2, f3 in zip(fs1, fs2, fs3):
            fs_diff.append(self.drop(paddle.concat([f1, f2, f3], axis=1)))
        y = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]

@manager.MODELS.add_component
class CX_Uper_3B_plus(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ckpt_path=None
                    ):
        super(CX_Uper_3B_plus, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
            self.backbone3 = convnext.convnext_tiny(in_chans=116)
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
            self.backbone3 = convnext.convnext_small(in_chans=116)
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
            self.backbone3 = convnext.convnext_base(in_chans=116)
        
        self.decode_head2b = UPerHead_2B(self.backbone1.dims[:3], num_classes=num_classes)
        self.decode_head3b = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)
        self.ckpt_path = ckpt_path
        self.load_ckpt()
        
    def load_ckpt(self):
        if self.ckpt_path is not None:
            para_state_dict = paddle.load(self.ckpt_path)
            self.set_state_dict(para_state_dict)
            print('load form:', self.ckpt_path)

    def forward(self, t1, t2):
        t3 = t2
        t2 = t1[:, 3:, ...]
        t1 = t1[:, :3, ...]
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            fs_diff.append(self.drop(paddle.concat([f1, f2], axis=1)))
        y2 = self.decode_head2b(fs_diff)
        y3 = self.decode_head3b(fs3)
        y = y2+y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone = convnext.convnext_small()
        else:
            self.backbone = convnext.convnext_base()
        
        self.decode_head = UPerHead(self.backbone.dims, num_classes=2)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone(t1)
        fs2 = self.backbone(t2)
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            fs_diff.append(self.drop(paddle.abs(f1 - f2)))
        y = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]
@manager.MODELS.add_component
class CX_Uper2(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper2, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone = convnext.convnext_tiny()
        else:
            self.backbone = convnext.convnext_small()
        
        self.decode_head = UPerHead2(self.backbone.dims, num_classes=2)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone(t1)
        fs2 = self.backbone(t2)
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            fs_diff.append(self.drop(paddle.abs(f1 - f2)))
        y, y_aug = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out, y_aug]

class UPerHead2(nn.Layer):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.LayerList()
        self.fpn_out = nn.LayerList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels)*channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.conv_seg = nn.Conv2D(channel, num_classes, 1)
        self.conv_seg_aug = nn.Conv2DTranspose(128, 2, kernel_size=2, stride=2)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
 
        output_feature = self.bottleneck(paddle.concat(fpn_features, axis=1))
        output_aug = self.conv_seg_aug(output_feature)
        output = self.conv_seg(self.dropout(output_feature))

        return output, output_aug

class UPerHead_2B(nn.Layer):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        in_channels = [ch*2 for ch in in_channels]
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.LayerList()
        self.fpn_out = nn.LayerList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels)*channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.conv_seg = nn.Conv2D(channel, num_classes, 1)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
 
        output = self.bottleneck(paddle.concat(fpn_features, axis=1))
        output = self.conv_seg(self.dropout(output))
        return output

class UPerHead_3B(nn.Layer):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        in_channels = [ch*3 for ch in in_channels]
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.LayerList()
        self.fpn_out = nn.LayerList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels)*channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.conv_seg = nn.Conv2D(channel, num_classes, 1)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
 
        output = self.bottleneck(paddle.concat(fpn_features, axis=1))
        output = self.conv_seg(self.dropout(output))
        return output


class UPerHead(nn.Layer):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.LayerList()
        self.fpn_out = nn.LayerList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels)*channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.conv_seg = nn.Conv2D(channel, num_classes, 1)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
 
        output = self.bottleneck(paddle.concat(fpn_features, axis=1))
        output = self.conv_seg(self.dropout(output))
        return output

class PPM(nn.Layer):
    """Pyramid Pooling Module in PSPNet
    """
    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.LayerList([
            nn.Sequential(
                nn.AdaptiveAvgPool2D(scale),
                ConvModule(c1, c2, 1)
            )
        for scale in scales])

        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for stage in self.stages:
            outs.append(F.interpolate(stage(x), size=x.shape[-2:], mode='bilinear', align_corners=True))

        outs = [x] + outs[::-1]
        out = self.bottleneck(paddle.concat(outs, axis=1))
        return out

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2D(c1, c2, k, s, p, d, g, bias_attr=False),
            nn.BatchNorm2D(c2),
            nn.ReLU(True)
        )