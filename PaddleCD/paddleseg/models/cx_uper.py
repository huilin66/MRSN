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
        
        self.decode_head4b = UPerHead_4B(self.backbone1.dims[:3], num_classes=num_classes)
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
        for f0, f1, f2, f3 in zip(fs0, fs1, fs2, fs3):
            f = self.drop(paddle.concat([f0, f1, f2, f3], axis=1))
            fs_diff.append(f)
        y = self.decode_head4b(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper_4B_CA_FLASH(CX_Uper_4B):
    def __init__(self, **kwargs):
        super(CX_Uper_4B_CA_FLASH, self).__init__(**kwargs)

        self.cas1 = nn.LayerList()
        self.cas2 = nn.LayerList()
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
            f = self.drop(paddle.concat([ff1, ff2], axis=1))
            fs_diff_ca.append(f)
        y = self.decode_head4b(fs_diff_ca)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]

@manager.MODELS.add_component
class CX_Uper_4B_CA_FLASH_V1(CX_Uper_4B):
    def __init__(self, **kwargs):
        super(CX_Uper_4B_CA_FLASH_V1, self).__init__(**kwargs)

        self.cas1 = nn.LayerList()
        self.cas2 = nn.LayerList()
        for dim in self.backbone1.dims[:3]:
            ca1 = FlashMultiHeadAttention(dim * 3, 8, kdim=dim, vdim=dim)
            ca2 = FlashMultiHeadAttention(dim, 8, kdim=dim * 3, vdim=dim * 3)
            self.cas1.append(ca1)
            self.cas2.append(ca2)

        self.alpha1 = self.create_parameter(
            shape=[3],
            default_initializer=nn.initializer.Constant(0.0))
        self.alpha2 = self.create_parameter(
            shape=[3],
            default_initializer=nn.initializer.Constant(0.0))

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
        for idx, (ca1, ca2, f1, f2) in enumerate(zip(self.cas1, self.cas2, fs_diff, fs3)):
            f1_ori = f1  # [B, dim*3, H, W]
            f2_ori = f2  # [B, dim,   H, W]
            shape1, shape2 = f1.shape, f2.shape
            f1 = f1.reshape((shape1[0], shape1[1], -1)).transpose((0, 2, 1))
            f2 = f2.reshape((shape2[0], shape2[1], -1)).transpose((0, 2, 1))
            attn1 = ca1(f1, f2, f2)
            attn1 = attn1.transpose((0, 2, 1)).reshape((shape1[0], shape1[1], shape1[2], shape1[3]))
            attn2 = ca2(f2, f1, f1)
            attn2 = attn2.transpose((0, 2, 1)).reshape((shape2[0], shape2[1], shape2[2], shape2[3]))
            # Residual + Learnable Gate
            ff1 = f1_ori + self.alpha1[idx] * attn1
            ff2 = f2_ori + self.alpha2[idx] * attn2
            f = self.drop(paddle.concat([ff1, ff2], axis=1))
            fs_diff_ca.append(f)
        y = self.decode_head4b(fs_diff_ca)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper_4B_CA_FLASH_V2(CX_Uper_4B):
    def __init__(self, **kwargs):
        super(CX_Uper_4B_CA_FLASH_V2, self).__init__(**kwargs)

        self.cas1 = nn.LayerList()
        self.cas2 = nn.LayerList()
        self.pos1 = nn.LayerList()  # pos enc for fs_diff
        self.pos2 = nn.LayerList()  # pos enc for fs3
        for dim in self.backbone1.dims[:3]:
            ca1 = FlashMultiHeadAttention(dim * 3, 8, kdim=dim, vdim=dim)
            ca2 = FlashMultiHeadAttention(dim, 8, kdim=dim * 3, vdim=dim * 3)
            self.cas1.append(ca1)
            self.cas2.append(ca2)
            self.pos1.append(PosEnhance(dim * 3))
            self.pos2.append(PosEnhance(dim))

        self.alpha1 = self.create_parameter(
            shape=[3],
            default_initializer=nn.initializer.Constant(0.0))
        self.alpha2 = self.create_parameter(
            shape=[3],
            default_initializer=nn.initializer.Constant(0.0))

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
        for idx, (ca1, ca2, pos1, pos2, f1, f2) in enumerate(
                zip(self.cas1, self.cas2, self.pos1, self.pos2, fs_diff, fs3)):
            f1_ori = f1
            f2_ori = f2
            # Position encoding before tokenization
            f1 = pos1(f1)
            f2 = pos2(f2)
            shape1, shape2 = f1.shape, f2.shape
            f1 = f1.reshape((shape1[0], shape1[1], -1)).transpose((0, 2, 1))
            f2 = f2.reshape((shape2[0], shape2[1], -1)).transpose((0, 2, 1))
            attn1 = ca1(f1, f2, f2)
            attn1 = attn1.transpose((0, 2, 1)).reshape((shape1[0], shape1[1], shape1[2], shape1[3]))
            attn2 = ca2(f2, f1, f1)
            attn2 = attn2.transpose((0, 2, 1)).reshape((shape2[0], shape2[1], shape2[2], shape2[3]))
            # Residual + Learnable Gate
            ff1 = f1_ori + self.alpha1[idx] * attn1
            ff2 = f2_ori + self.alpha2[idx] * attn2
            f = self.drop(paddle.concat([ff1, ff2], axis=1))
            fs_diff_ca.append(f)
        y = self.decode_head4b(fs_diff_ca)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]



@manager.MODELS.add_component
class CX_Uper_4B_CA_FLASH_V3(CX_Uper_4B):
    def __init__(self, **kwargs):
        super(CX_Uper_4B_CA_FLASH_V3, self).__init__(**kwargs)

        # Only one CA pair for the last (highest) stage
        dim_last = self.backbone1.dims[2]
        self.ca1 = FlashMultiHeadAttention(dim_last * 3, 8, kdim=dim_last, vdim=dim_last)
        self.ca2 = FlashMultiHeadAttention(dim_last, 8, kdim=dim_last * 3, vdim=dim_last * 3)

        self.alpha1 = self.create_parameter(
            shape=[1],
            default_initializer=nn.initializer.Constant(0.0))
        self.alpha2 = self.create_parameter(
            shape=[1],
            default_initializer=nn.initializer.Constant(0.0))

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
        for idx, (f1, f2) in enumerate(zip(fs_diff, fs3)):
            if idx < 2:
                # Stage 0,1: direct concat (same as CX_Uper_4B)
                f = self.drop(paddle.concat([f1, f2], axis=1))
                fs_diff_ca.append(f)
            else:
                # Stage 2 (highest): Cross-Attention + Residual + Gate
                f1_ori = f1
                f2_ori = f2
                shape1, shape2 = f1.shape, f2.shape
                f1_t = f1.reshape((shape1[0], shape1[1], -1)).transpose((0, 2, 1))
                f2_t = f2.reshape((shape2[0], shape2[1], -1)).transpose((0, 2, 1))
                attn1 = self.ca1(f1_t, f2_t, f2_t)
                attn1 = attn1.transpose((0, 2, 1)).reshape((shape1[0], shape1[1], shape1[2], shape1[3]))
                attn2 = self.ca2(f2_t, f1_t, f1_t)
                attn2 = attn2.transpose((0, 2, 1)).reshape((shape2[0], shape2[1], shape2[2], shape2[3]))
                ff1 = f1_ori + self.alpha1[0] * attn1
                ff2 = f2_ori + self.alpha2[0] * attn2
                f = self.drop(paddle.concat([ff1, ff2], axis=1))
                fs_diff_ca.append(f)
        y = self.decode_head4b(fs_diff_ca)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]



@manager.MODELS.add_component
class CX_Uper_4B_CA_FLASH_V4(CX_Uper_4B):
    def __init__(self, **kwargs):
        super(CX_Uper_4B_CA_FLASH_V4, self).__init__(**kwargs)

        # Only last stage: Cross-Attention + PosEnhance
        dim_last = self.backbone1.dims[2]
        self.ca1 = FlashMultiHeadAttention(dim_last * 3, 8, kdim=dim_last, vdim=dim_last)
        self.ca2 = FlashMultiHeadAttention(dim_last, 8, kdim=dim_last * 3, vdim=dim_last * 3)
        self.pos1 = PosEnhance(dim_last * 3)
        self.pos2 = PosEnhance(dim_last)

        self.alpha1 = self.create_parameter(
            shape=[1],
            default_initializer=nn.initializer.Constant(0.0))
        self.alpha2 = self.create_parameter(
            shape=[1],
            default_initializer=nn.initializer.Constant(0.0))

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
        for idx, (f1, f2) in enumerate(zip(fs_diff, fs3)):
            if idx < 2:
                # Stage 0,1: direct concat
                f = self.drop(paddle.concat([f1, f2], axis=1))
                fs_diff_ca.append(f)
            else:
                # Stage 2: PosEnhance + Cross-Attention + Residual + Gate
                f1_ori = f1
                f2_ori = f2
                # DWConv position encoding
                f1 = self.pos1(f1)
                f2 = self.pos2(f2)
                shape1, shape2 = f1.shape, f2.shape
                f1_t = f1.reshape((shape1[0], shape1[1], -1)).transpose((0, 2, 1))
                f2_t = f2.reshape((shape2[0], shape2[1], -1)).transpose((0, 2, 1))
                attn1 = self.ca1(f1_t, f2_t, f2_t)
                attn1 = attn1.transpose((0, 2, 1)).reshape((shape1[0], shape1[1], shape1[2], shape1[3]))
                attn2 = self.ca2(f2_t, f1_t, f1_t)
                attn2 = attn2.transpose((0, 2, 1)).reshape((shape2[0], shape2[1], shape2[2], shape2[3]))
                ff1 = f1_ori + self.alpha1[0] * attn1
                ff2 = f2_ori + self.alpha2[0] * attn2
                f = self.drop(paddle.concat([ff1, ff2], axis=1))
                fs_diff_ca.append(f)
        y = self.decode_head4b(fs_diff_ca)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


# ============================================================
# V5: Spatial Attention (f0+f1+f2) + Channel Attention (fs3)
# RGB fused features → spatial attention (where to look)
# HSI features        → channel attention (which bands matter)
# Residual + Learnable Gate, alpha=0 初始化
# ============================================================
@manager.MODELS.add_component
class CX_Uper_4B_CA_FLASH_V5(CX_Uper_4B):
    def __init__(self, **kwargs):
        super(CX_Uper_4B_CA_FLASH_V5, self).__init__(**kwargs)

        self.spatial_attn = nn.LayerList()
        self.channel_attn = nn.LayerList()
        for dim in self.backbone1.dims[:3]:
            # Spatial attention on fused RGB+DSM+NIR (dim*3 channels)
            self.spatial_attn.append(SpatialAttention())
            # Channel attention on HSI (dim channels)
            self.channel_attn.append(ChannelAttention(dim))

        self.alpha1 = self.create_parameter(
            shape=[3],
            default_initializer=nn.initializer.Constant(0.0))
        self.alpha2 = self.create_parameter(
            shape=[3],
            default_initializer=nn.initializer.Constant(0.0))

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
        for idx, (sp_attn, ch_attn, f1, f2) in enumerate(
                zip(self.spatial_attn, self.channel_attn, fs_diff, fs3)):
            f1_ori = f1  # [B, dim*3, H, W] — fused RGB+DSM+NIR
            f2_ori = f2  # [B, dim,   H, W] — HSI
            # Spatial attention: learns "where" in the fused RGB features
            sp_out = sp_attn(f1)
            # Channel attention: learns "which bands" in HSI features
            ch_out = ch_attn(f2)
            # Residual + Learnable Gate
            ff1 = f1_ori + self.alpha1[idx] * sp_out
            ff2 = f2_ori + self.alpha2[idx] * ch_out
            f = self.drop(paddle.concat([ff1, ff2], axis=1))
            fs_diff_ca.append(f)
        y = self.decode_head4b(fs_diff_ca)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]

@manager.MODELS.add_component
class CX_Uper_4B2H(nn.Layer):
    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    hsi_chs=242, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper_4B2H, self).__init__()
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
        

        self.decode_head3b = UPerHead_3B(self.backbone1.dims[:3], num_classes=num_classes)
        self.decode_head1b = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
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
        y2 = self.decode_head3b(fs_diff)
        y3 = self.decode_head1b(fs3)
        y = y2+y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper_4B2H_CA_FLASH(CX_Uper_4B2H):
    def __init__(self, **kwargs):
        super(CX_Uper_4B2H_CA_FLASH, self).__init__(**kwargs)

        self.cas1 = nn.LayerList()
        self.cas2 = nn.LayerList()
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
            f1 = f1.reshape((shape1[0], shape1[1], -1)).transpose((0, 2, 1))
            f2 = f2.reshape((shape2[0], shape2[1], -1)).transpose((0, 2, 1))
            ff1 = ca1(f1, f2, f2)
            ff1 = ff1.transpose((0, 2, 1)).reshape((shape1[0], shape1[1], shape1[2], shape1[3]))
            ff2 = ca2(f2, f1, f1)
            ff2 = ff2.transpose((0, 2, 1)).reshape((shape2[0], shape2[1], shape2[2], shape2[3]))
            fs_diff_ca.append(ff1)
            fs3_ca.append(ff2)
        y2 = self.decode_head3b(fs_diff_ca)
        y3 = self.decode_head1b(fs3_ca)
        y = y2+y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


# region head
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

class UPerHead_4B(nn.Layer):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        in_channels = [ch*4 for ch in in_channels]
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


class PosEnhance(nn.Layer):
    """DWConv position encoding for attention.

    Adds spatial structure awareness before tokenization.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.dwconv(x)


class SpatialAttention(nn.Layer):
    """Spatial attention: generates H*W attention map via avg+max pooling + conv.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]  (same shape, spatially reweighted)
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2D(2, 1, kernel_size, padding=kernel_size // 2, bias_attr=False)

    def forward(self, x):
        avg_out = x.mean(axis=1, keepdim=True)   # [B, 1, H, W]
        max_out = x.max(axis=1, keepdim=True)    # [B, 1, H, W]
        attn = F.sigmoid(self.conv(paddle.concat([avg_out, max_out], axis=1)))  # [B, 1, H, W]
        return x * attn


class ChannelAttention(nn.Layer):
    """Channel attention (SE-style): reweights channels via GAP + FC + sigmoid.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]  (same shape, channel-reweighted)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.mean(axis=[2, 3])                  # [B, C]
        y = self.fc(y).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * F.sigmoid(y)

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

# endregion