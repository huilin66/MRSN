"""PaddlePaddle/PaddleSeg implementation of HighDAN.

Input convention follows the provided CX_Uper models:
    t1: [B, 6, H, W], first 4 channels are MSI and last 2 are SAR.
    t2: [B, hsi_chs, H, W], HSI input.

The default ``forward(t1, t2)`` is compatible with PaddleSeg and returns
``[logits]``. The original domain-adaptation path is retained through
``forward_adaptation`` and the converted discriminator classes at the bottom
of this file.
"""

import logging
from typing import List, Optional, Sequence, Tuple, Union

import paddle
from paddle import Tensor, nn
from paddle.nn import functional as F

from paddleseg.cvlibs import manager
from paddleseg.models.backbones.hrnet import HRNet_W48


# PyTorch BatchNorm momentum=0.01 means:
# running = 0.99 * running + 0.01 * batch.
# Paddle's momentum is the coefficient of the old running statistic.
BN_MOMENTUM = 0.99
logger = logging.getLogger(__name__)


def _initialize_weights(layer: nn.Layer) -> None:
    """Initialize convolution and normalization layers without external utils."""
    kaiming = nn.initializer.KaimingNormal()
    zeros = nn.initializer.Constant(0.0)
    ones = nn.initializer.Constant(1.0)

    for sublayer in layer.sublayers():
        if isinstance(sublayer, (nn.Conv2D, nn.Conv2DTranspose)):
            if sublayer.weight is not None:
                kaiming(sublayer.weight)
            if sublayer.bias is not None:
                zeros(sublayer.bias)
        elif isinstance(sublayer, nn.BatchNorm2D):
            if sublayer.weight is not None:
                ones(sublayer.weight)
            if sublayer.bias is not None:
                zeros(sublayer.bias)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2D:
    """3x3 convolution with padding."""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias_attr=False,
    )


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Layer] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Layer] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False,
        )
        self.bn2 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2D(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias_attr=False,
        )
        self.bn3 = nn.BatchNorm2D(
            planes * self.expansion,
            momentum=BN_MOMENTUM,
        )
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


class HighResolutionModule(nn.Layer):
    def __init__(
        self,
        num_branches: int,
        block,
        num_blocks: Sequence[int],
        num_inchannels: List[int],
        num_channels: Sequence[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ) -> None:
        super().__init__()
        self._check_branches(
            num_branches,
            num_blocks,
            num_inchannels,
            num_channels,
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches,
            block,
            num_blocks,
            num_channels,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

    @staticmethod
    def _check_branches(
        num_branches: int,
        num_blocks: Sequence[int],
        num_inchannels: Sequence[int],
        num_channels: Sequence[int],
    ) -> None:
        if num_branches != len(num_blocks):
            raise ValueError(
                f"NUM_BRANCHES({num_branches}) <> NUM_BLOCKS({len(num_blocks)})"
            )
        if num_branches != len(num_channels):
            raise ValueError(
                f"NUM_BRANCHES({num_branches}) <> NUM_CHANNELS({len(num_channels)})"
            )
        if num_branches != len(num_inchannels):
            raise ValueError(
                f"NUM_BRANCHES({num_branches}) <> NUM_INCHANNELS({len(num_inchannels)})"
            )

    def _make_one_branch(
        self,
        branch_index: int,
        block,
        num_blocks: Sequence[int],
        num_channels: Sequence[int],
        stride: int = 1,
    ) -> nn.Sequential:
        expected_channels = num_channels[branch_index] * block.expansion
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != expected_channels:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.num_inchannels[branch_index],
                    expected_channels,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(expected_channels, momentum=BN_MOMENTUM),
            )

        layers = [
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        ]
        self.num_inchannels[branch_index] = expected_channels

        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block,
        num_blocks: Sequence[int],
        num_channels: Sequence[int],
    ) -> nn.LayerList:
        branches = [
            self._make_one_branch(i, block, num_blocks, num_channels)
            for i in range(num_branches)
        ]
        return nn.LayerList(branches)

    def _make_fuse_layers(self) -> Optional[nn.LayerList]:
        if self.num_branches == 1:
            return None

        fuse_layers = []
        output_branches = self.num_branches if self.multi_scale_output else 1

        for i in range(output_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2D(
                                self.num_inchannels[j],
                                self.num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias_attr=False,
                            ),
                            nn.BatchNorm2D(
                                self.num_inchannels[i],
                                momentum=BN_MOMENTUM,
                            ),
                        )
                    )
                elif j == i:
                    # Paddle LayerList stores Layer objects rather than None.
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    in_channels = self.num_inchannels[j]
                    for k in range(i - j):
                        is_last = k == i - j - 1
                        out_channels = (
                            self.num_inchannels[i] if is_last else in_channels
                        )
                        layers = [
                            nn.Conv2D(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias_attr=False,
                            ),
                            nn.BatchNorm2D(
                                out_channels,
                                momentum=BN_MOMENTUM,
                            ),
                        ]
                        if not is_last:
                            layers.append(nn.ReLU())
                        conv3x3s.append(nn.Sequential(*layers))
                        in_channels = out_channels
                    fuse_layer.append(nn.Sequential(*conv3x3s))

            fuse_layers.append(nn.LayerList(fuse_layer))

        return nn.LayerList(fuse_layers)

    def get_num_inchannels(self) -> List[int]:
        return self.num_inchannels

    def forward(self, x: Sequence[Tensor]) -> List[Tensor]:
        x = list(x)
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            if i == 0:
                y = x[0]
            else:
                y = self.fuse_layers[i][0](x[0])

            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    projected = self.fuse_layers[i][j](x[j])
                    projected = F.interpolate(
                        projected,
                        size=x[i].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    y = y + projected
                else:
                    y = y + self.fuse_layers[i][j](x[j])

            x_fuse.append(self.relu(y))

        return x_fuse


BLOCKS_DICT = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck,
}


@manager.MODELS.add_component
class HighDAN(nn.Layer):
    """High-resolution domain-adaptation network for HSI/MSI/SAR segmentation.

    Args:
        in_channels: Kept for compatibility with existing PaddleSeg YAML files.
            The actual first input contains ``msi_chs + sar_chs`` channels.
        num_classes: Number of semantic classes.
        hsi_chs: Number of HSI bands in ``t2``.
        msi_chs: Number of MSI channels at the beginning of ``t1``.
        sar_chs: Number of SAR channels at the end of ``t1``.
        align_corners: Interpolation setting for final logits.
        pretrained: Local path or URL of PaddleSeg's ImageNet-pretrained
            HRNet-W48 ``.pdparams`` file. Only the compatible HRNet body is
            transferred; modality-specific stems and the segmentation head
            remain randomly initialized.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hsi_chs: int = 242,
        msi_chs: int = 4,
        sar_chs: int = 2,
        align_corners: bool = True,
        pretrained: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hsi_chs = hsi_chs
        self.msi_chs = msi_chs
        self.sar_chs = sar_chs
        self.align_corners = align_corners
        self.pretrained = pretrained

        # HSI stem.
        self.conv1 = nn.Conv2D(
            hsi_chs,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=False,
        )
        self.bn1 = nn.BatchNorm2D(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2D(
            64,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
        )
        self.bn2 = nn.BatchNorm2D(64, momentum=BN_MOMENTUM)

        # MSI and SAR stems.
        self.conv_msi = nn.Conv2D(
            msi_chs,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=False,
        )
        self.bn_msi = nn.BatchNorm2D(64, momentum=BN_MOMENTUM)
        self.conv_sar = nn.Conv2D(
            sar_chs,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=False,
        )
        self.bn_sar = nn.BatchNorm2D(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

        # The three modality-specific stage-1 networks are independent.
        block = BLOCKS_DICT["BOTTLENECK"]
        self.layer1 = self._make_layer(block, 64, 64, blocks=4)
        self.msi_layer1 = self._make_layer(block, 64, 64, blocks=4)
        self.sar_layer1 = self._make_layer(block, 64, 64, blocks=4)
        stage1_out_channel = block.expansion * 64

        # Later HRNet stages are shared across the three modalities, matching
        # the original PyTorch implementation.
        num_channels = [48, 96]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel],
            num_channels,
        )
        self.stage2_num_branches = 2
        self.stage2, pre_stage_channels = self._make_stage(
            num_modules=1,
            num_branches=2,
            num_blocks=[4, 4],
            num_channels=[48, 96],
            block_name="BASIC",
            fuse_method="SUM",
            num_inchannels=num_channels,
        )

        num_channels = [48, 96, 192]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels,
            num_channels,
        )
        self.stage3_num_branches = 3
        self.stage3, pre_stage_channels = self._make_stage(
            num_modules=1,
            num_branches=3,
            num_blocks=[3, 3, 3],
            num_channels=[48, 96, 192],
            block_name="BASIC",
            fuse_method="SUM",
            num_inchannels=num_channels,
        )

        num_channels = [48, 96, 192, 384]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels,
            num_channels,
        )
        self.stage4_num_branches = 4
        self.stage4, pre_stage_channels = self._make_stage(
            num_modules=1,
            num_branches=4,
            num_blocks=[1, 1, 1, 1],
            num_channels=[48, 96, 192, 384],
            block_name="BASIC",
            fuse_method="SUM",
            num_inchannels=num_channels,
            multi_scale_output=True,
        )

        # (48 + 96 + 192 + 384) * 3 modalities = 2160 channels.
        self.feature_channels = sum(pre_stage_channels) * 3
        self.last_layer = nn.Sequential(
            nn.Conv2D(self.feature_channels, 256, 3, stride=1, padding=1),
            nn.BatchNorm2D(256, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2D(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2D(128, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2D(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2D(64, momentum=BN_MOMENTUM),
            nn.ReLU(),
        )
        self.transconv = nn.Conv2DTranspose(
            64,
            64,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        self.final_conv = nn.Conv2D(64, num_classes, kernel_size=1, stride=1)
        self.tanh = nn.Tanh()

        _initialize_weights(self)
        if self.pretrained is not None:
            self._load_hrnet_w48_pretrained(self.pretrained)

    @staticmethod
    def _copy_ordered_state(
        source: nn.Layer,
        target: nn.Layer,
        label: str,
    ) -> int:
        """Copy two structurally equivalent Paddle layers by state order.

        PaddleSeg HRNet wraps Conv/BN pairs in ``ConvBN`` or ``ConvBNReLU``,
        while HighDAN stores them as explicit ``Conv2D`` and ``BatchNorm2D``
        layers. Their state keys therefore differ, but the state registration
        order and tensor shapes are equivalent for matched blocks.
        """
        source_items = list(source.state_dict().items())
        target_state = target.state_dict()
        target_items = list(target_state.items())

        if len(source_items) != len(target_items):
            raise RuntimeError(
                f"Cannot transfer {label}: source has {len(source_items)} "
                f"state tensors, target has {len(target_items)}."
            )

        for (source_key, source_value), (target_key, target_value) in zip(
            source_items, target_items
        ):
            if list(source_value.shape) != list(target_value.shape):
                raise RuntimeError(
                    f"Cannot transfer {label}: {source_key} "
                    f"{list(source_value.shape)} does not match {target_key} "
                    f"{list(target_value.shape)}."
                )
            target_state[target_key] = source_value

        target.set_state_dict(target_state)
        return len(target_items)

    def _copy_transition_from_paddleseg(
        self,
        source_transition: nn.Layer,
        target_transition: nn.LayerList,
        label: str,
    ) -> int:
        loaded = 0
        source_layers = source_transition.conv_bn_func_list

        if len(source_layers) != len(target_transition):
            raise RuntimeError(
                f"Cannot transfer {label}: source has {len(source_layers)} "
                f"branches, target has {len(target_transition)}."
            )

        for branch_index, (source_layer, target_layer) in enumerate(
            zip(source_layers, target_transition)
        ):
            source_count = (
                0 if source_layer is None else len(source_layer.state_dict())
            )
            target_count = len(target_layer.state_dict())

            if source_count == 0 and target_count == 0:
                continue
            if source_layer is None:
                raise RuntimeError(
                    f"Cannot transfer {label}[{branch_index}]: source is "
                    "identity but target contains parameters."
                )

            loaded += self._copy_ordered_state(
                source_layer,
                target_layer,
                f"{label}[{branch_index}]",
            )
        return loaded

    def _copy_stage_module_from_paddleseg(
        self,
        source_module: nn.Layer,
        target_module: HighResolutionModule,
        label: str,
    ) -> int:
        """Transfer the first standard HRNet module into a HighDAN stage.

        HighDAN keeps only one module in Stage3 and Stage4 and uses fewer
        residual blocks. We therefore copy exactly the blocks present in the
        target and then copy the structurally matching fusion layers.
        """
        loaded = 0
        source_branches = source_module.branches_func.basic_block_list
        target_branches = target_module.branches

        if len(source_branches) != len(target_branches):
            raise RuntimeError(
                f"Cannot transfer {label}: source has {len(source_branches)} "
                f"branches, target has {len(target_branches)}."
            )

        for branch_index, target_branch in enumerate(target_branches):
            source_branch = source_branches[branch_index]
            if len(source_branch) < len(target_branch):
                raise RuntimeError(
                    f"Cannot transfer {label}.branch{branch_index}: source "
                    f"has {len(source_branch)} blocks, target requires "
                    f"{len(target_branch)}."
                )

            for block_index, target_block in enumerate(target_branch):
                loaded += self._copy_ordered_state(
                    source_branch[block_index],
                    target_block,
                    f"{label}.branch{branch_index}.block{block_index}",
                )

        if target_module.fuse_layers is not None:
            loaded += self._copy_ordered_state(
                source_module.fuse_func,
                target_module.fuse_layers,
                f"{label}.fuse",
            )
        return loaded

    def _load_hrnet_w48_pretrained(self, pretrained: str) -> None:
        """Load PaddleSeg ImageNet HRNet-W48 weights into HighDAN.

        The transfer follows the original HighDAN training strategy:

        * skip the RGB stem because HSI/MSI/SAR channel counts differ;
        * initialize the HSI Stage1 from HRNet and copy it to MSI/SAR;
        * load Transition1-3 and the first compatible Stage2-4 module;
        * load only the residual blocks that exist in the shallower HighDAN;
        * leave ``last_layer``, ``transconv`` and ``final_conv`` untouched.

        Args:
            pretrained: PaddleSeg HRNet-W48 ImageNet weight path or URL.
        """
        logger.info(
            "Loading PaddleSeg HRNet-W48 pretrained weights for HighDAN "
            "from %s",
            pretrained,
        )

        try:
            reference = HRNet_W48(pretrained=pretrained)
        except Exception as exc:
            raise RuntimeError(
                "Failed to construct PaddleSeg HRNet_W48 with pretrained="
                f"{pretrained!r}. Ensure this is a Paddle .pdparams backbone "
                "checkpoint or a valid PaddleSeg pretrained URL."
            ) from exc

        loaded = 0

        # Stage1: four bottleneck blocks. The same pretrained Stage1 is used
        # to initialize all three modality-specific branches.
        source_layer1 = reference.la1.bottleneck_block_list
        if len(source_layer1) < len(self.layer1):
            raise RuntimeError(
                "PaddleSeg HRNet-W48 Stage1 is shallower than HighDAN Stage1."
            )
        for block_index, target_block in enumerate(self.layer1):
            loaded += self._copy_ordered_state(
                source_layer1[block_index],
                target_block,
                f"layer1.block{block_index}",
            )

        # set_state_dict copies values into independent parameters; it does not
        # make the three modality branches share Parameter objects.
        self.msi_layer1.set_state_dict(self.layer1.state_dict())
        self.sar_layer1.set_state_dict(self.layer1.state_dict())
        loaded += 2 * len(self.layer1.state_dict())

        loaded += self._copy_transition_from_paddleseg(
            reference.tr1, self.transition1, "transition1"
        )
        loaded += self._copy_stage_module_from_paddleseg(
            reference.st2.stage_func_list[0], self.stage2[0], "stage2.module0"
        )
        loaded += self._copy_transition_from_paddleseg(
            reference.tr2, self.transition2, "transition2"
        )
        loaded += self._copy_stage_module_from_paddleseg(
            reference.st3.stage_func_list[0], self.stage3[0], "stage3.module0"
        )
        loaded += self._copy_transition_from_paddleseg(
            reference.tr3, self.transition3, "transition3"
        )
        loaded += self._copy_stage_module_from_paddleseg(
            reference.st4.stage_func_list[0], self.stage4[0], "stage4.module0"
        )

        logger.info(
            "Loaded %d HighDAN state tensors from PaddleSeg HRNet-W48. "
            "Skipped modality stems and segmentation head.",
            loaded,
        )
        del reference

    def _make_transition_layer(
        self,
        num_channels_pre_layer: Sequence[int],
        num_channels_cur_layer: Sequence[int],
    ) -> nn.LayerList:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []

        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2D(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias_attr=False,
                            ),
                            nn.BatchNorm2D(
                                num_channels_cur_layer[i],
                                momentum=BN_MOMENTUM,
                            ),
                            nn.ReLU(),
                        )
                    )
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                in_channels = num_channels_pre_layer[-1]
                for j in range(i + 1 - num_branches_pre):
                    is_last = j == i - num_branches_pre
                    out_channels = (
                        num_channels_cur_layer[i] if is_last else in_channels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2D(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias_attr=False,
                            ),
                            nn.BatchNorm2D(
                                out_channels,
                                momentum=BN_MOMENTUM,
                            ),
                            nn.ReLU(),
                        )
                    )
                    in_channels = out_channels
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.LayerList(transition_layers)

    @staticmethod
    def _make_layer(
        block,
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        out_channels = planes * block.expansion
        if stride != 1 or inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2D(
                    inplanes,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(out_channels, momentum=BN_MOMENTUM),
            )

        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_stage(
        num_modules: int,
        num_branches: int,
        num_blocks: Sequence[int],
        num_channels: Sequence[int],
        block_name: str,
        fuse_method: str,
        num_inchannels: List[int],
        multi_scale_output: bool = True,
    ) -> Tuple[nn.Sequential, List[int]]:
        block = BLOCKS_DICT[block_name]
        modules = []

        for i in range(num_modules):
            reset_multi_scale_output = not (
                not multi_scale_output and i == num_modules - 1
            )
            module = HighResolutionModule(
                num_branches,
                block,
                num_blocks,
                num_inchannels,
                num_channels,
                fuse_method,
                reset_multi_scale_output,
            )
            modules.append(module)
            num_inchannels = module.get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    @staticmethod
    def _transition_forward(
        features: Sequence[Tensor],
        transitions: nn.LayerList,
        num_branches: int,
    ) -> List[Tensor]:
        """Apply an HRNet transition to one modality's branch list."""
        outputs = []
        old_branches = len(features)
        for i in range(num_branches):
            source = features[i] if i < old_branches else features[-1]
            outputs.append(transitions[i](source))
        return outputs

    def _forward_features(
        self,
        hsi: Tensor,
        msi: Tensor,
        sar: Tensor,
    ) -> Tensor:
        hsi = self.relu(self.bn1(self.conv1(hsi)))
        hsi = self.relu(self.bn2(self.conv2(hsi)))
        msi = self.relu(self.bn_msi(self.conv_msi(msi)))
        sar = self.relu(self.bn_sar(self.conv_sar(sar)))

        hsi = self.layer1(hsi)
        msi = self.msi_layer1(msi)
        sar = self.sar_layer1(sar)

        hsi = self.stage2(
            self._transition_forward([hsi], self.transition1, self.stage2_num_branches)
        )
        msi = self.stage2(
            self._transition_forward([msi], self.transition1, self.stage2_num_branches)
        )
        sar = self.stage2(
            self._transition_forward([sar], self.transition1, self.stage2_num_branches)
        )

        hsi = self.stage3(
            self._transition_forward(hsi, self.transition2, self.stage3_num_branches)
        )
        msi = self.stage3(
            self._transition_forward(msi, self.transition2, self.stage3_num_branches)
        )
        sar = self.stage3(
            self._transition_forward(sar, self.transition2, self.stage3_num_branches)
        )

        hsi = self.stage4(
            self._transition_forward(hsi, self.transition3, self.stage4_num_branches)
        )
        msi = self.stage4(
            self._transition_forward(msi, self.transition3, self.stage4_num_branches)
        )
        sar = self.stage4(
            self._transition_forward(sar, self.transition3, self.stage4_num_branches)
        )

        target_size = hsi[0].shape[-2:]
        modality_features = []
        for branches in (hsi, msi, sar):
            for branch_index, branch in enumerate(branches):
                if branch_index == 0:
                    modality_features.append(branch)
                else:
                    modality_features.append(
                        F.interpolate(
                            branch,
                            size=target_size,
                            mode="bilinear",
                            align_corners=True,
                        )
                    )

        return paddle.concat(modality_features, axis=1)

    def _decode(self, features: Tensor, output_size) -> Tensor:
        logits = self.last_layer(features)
        logits = self.transconv(logits)
        logits = self.final_conv(logits)
        # Guarantees exact PaddleSeg label resolution for odd input sizes too.
        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return logits

    def _split_inputs(self, t1: Tensor, t2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        expected_t1_channels = self.msi_chs + self.sar_chs
        if t1.shape[1] not in (None, -1, expected_t1_channels):
            raise ValueError(
                f"HighDAN expects t1 to have {expected_t1_channels} channels "
                f"({self.msi_chs} MSI + {self.sar_chs} SAR), got {t1.shape[1]}."
            )
        if t2.shape[1] not in (None, -1, self.hsi_chs):
            raise ValueError(
                f"HighDAN expects t2 to have {self.hsi_chs} HSI channels, "
                f"got {t2.shape[1]}."
            )

        msi = t1[:, : self.msi_chs, ...]
        sar = t1[:, self.msi_chs : expected_t1_channels, ...]
        hsi = t2
        return hsi, msi, sar

    def forward(self, t1: Tensor, t2: Tensor) -> List[Tensor]:
        """PaddleSeg forward: return a list containing full-resolution logits."""
        output_size = paddle.shape(t1)[2:]
        hsi, msi, sar = self._split_inputs(t1, t2)
        features = self._forward_features(hsi, msi, sar)
        logits = self._decode(features, output_size)
        return [logits]

    def forward_adaptation(
        self,
        t1: Tensor,
        t2: Tensor,
        discriminators: Optional[
            Union[nn.Layer, Sequence[nn.Layer], nn.LayerList]
        ] = None,
        domain: str = "source",
    ) -> Tuple[Tensor, Tensor]:
        """Original HighDAN-style domain path.

        Returns:
            ``(features, logits)`` rather than the PaddleSeg ``[logits]`` form.

        For ``domain='target'``, the first supplied discriminator produces the
        one-channel spatial modulation map used by the original implementation.
        """
        if domain not in {"source", "target"}:
            raise ValueError("domain must be either 'source' or 'target'.")

        output_size = paddle.shape(t1)[2:]
        hsi, msi, sar = self._split_inputs(t1, t2)
        features = self._forward_features(hsi, msi, sar)
        adapted_features = features

        if domain == "target":
            if discriminators is None:
                raise ValueError(
                    "A feature discriminator is required for domain='target'."
                )
            if isinstance(discriminators, (list, tuple, nn.LayerList)):
                if len(discriminators) == 0:
                    raise ValueError("discriminators cannot be empty.")
                discriminator = discriminators[0]
            else:
                discriminator = discriminators

            attention = paddle.abs(self.tanh(discriminator(features)))
            # [B, 1, H, W] broadcasts over the feature-channel dimension.
            adapted_features = features * (1.0 + attention)

        logits = self._decode(adapted_features, output_size)
        return features, logits


class FCDiscriminator(nn.Layer):
    """Paddle conversion of the original full-resolution discriminator."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 8:
            raise ValueError("num_classes must be at least 8.")

        self.conv1 = nn.Conv2D(num_classes, num_classes // 2, 3, 1, 1)
        self.conv2 = nn.Conv2D(num_classes // 2, num_classes // 4, 3, 1, 1)
        self.conv3 = nn.Conv2D(num_classes // 4, num_classes // 8, 3, 1, 1)
        self.classifier = nn.Conv2D(num_classes // 8, 1, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        _initialize_weights(self)

    def forward(self, x: Tensor) -> Tensor:
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        return self.classifier(x)


class OutspaceDiscriminator(nn.Layer):
    """Paddle conversion of the original strided output-space discriminator."""

    def __init__(self, num_classes: int, ndf: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2D(num_classes, ndf, 4, 2, 1)
        self.conv2 = nn.Conv2D(ndf, ndf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2D(ndf * 2, ndf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2D(ndf * 4, ndf * 8, 4, 2, 1)
        self.classifier = nn.Conv2D(ndf * 8, 1, 4, 2, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        _initialize_weights(self)

    def forward(self, x: Tensor) -> Tensor:
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        return self.classifier(x)


# Compatibility alias for code that imports the original class name directly.
HighResolutionNet = HighDAN
