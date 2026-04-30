import paddle
from paddle import nn, Tensor
from paddle.nn import functional as F
from typing import Tuple

from paddleseg.cvlibs import manager
from paddleseg.models.backbones import convnext



@manager.MODELS.add_component
class CX_Uper(nn.Layer):
    def __init__(self,
                 in_channels,
                 num_classes,
                 backb,
                 hsi_chs=242,
                 dropout_rate=0.0,
                 ):
        super(CX_Uper, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone = convnext.convnext_tiny(in_chans=hsi_chs + 6)
        elif backb == 'convnext_small':
            self.backbone = convnext.convnext_small(in_chans=hsi_chs + 6)
        else:
            self.backbone = convnext.convnext_base(in_chans=hsi_chs + 6)

        self.decode_head = UPerHead(self.backbone.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        t = paddle.concat([t1, t2], axis=1)
        fs = self.backbone(t)
        y = self.decode_head(fs)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper_2B(nn.Layer):
    def __init__(self,
                 in_channels,
                 num_classes,
                 backb,
                 hsi_chs=242,
                 dropout_rate=0.0,
                 ):
        super(CX_Uper_2B, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny(in_chans=6)
            self.backbone2 = convnext.convnext_tiny(in_chans=hsi_chs)
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small(in_chans=6)
            self.backbone2 = convnext.convnext_small(in_chans=hsi_chs)
        else:
            self.backbone1 = convnext.convnext_base(in_chans=6)
            self.backbone2 = convnext.convnext_base(in_chans=hsi_chs)

        self.decode_head = UPerHead_2B(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            fs_diff.append(self.drop(paddle.concat([f1, f2], axis=1)))
        y = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper_3B(nn.Layer):
    def __init__(self,
                 in_channels,
                 num_classes,
                 backb,
                 hsi_chs=242,
                 dropout_rate=0.0,
                 ):
        super(CX_Uper_3B, self).__init__()
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

        self.decode_head = UPerHead_3B(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        t3 = t2
        t2 = t1[:, 4:, ...]
        t1 = t1[:, :4, ...]
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f1, f2, f3 in zip(fs1, fs2, fs3):
            f = self.drop(paddle.concat([f1, f2, f3], axis=1))
            fs_diff.append(f)
        y = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper_4B(nn.Layer):
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


# region tools

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

        for in_ch in in_channels[:-1]:  # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels) * channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.conv_seg = nn.Conv2D(channel, num_classes, 1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features) - 1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear',
                                            align_corners=False)

        output = self.bottleneck(paddle.concat(fpn_features, axis=1))
        output = self.conv_seg(self.dropout(output))
        return output

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

# endregion