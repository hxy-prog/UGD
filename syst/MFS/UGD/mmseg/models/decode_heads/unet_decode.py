import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16
import torch
import warnings
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence,
                                         build_positional_encoding)
from torch.nn.init import normal_
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,
                      build_norm_layer)
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.ops import Upsample
from ..builder import BACKBONES
from ..utils import UpConvBlock


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super(BasicConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


@HEADS.register_module()
class Unetdecode(BaseDecodeHead):
    def __init__(self,
                    in_channels=256,
                    base_channels=64,
                    num_stages=5,
                    strides=(1, 1, 1, 1, 1),
                    enc_num_convs=(2, 2, 2, 2, 2),
                    dec_num_convs=(2, 2, 2, 2),
                    downsamples=(True, True, True, True),
                    enc_dilations=(1, 1, 1, 1, 1),
                    dec_dilations=(1, 1, 1, 1),
                    with_cp=False,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU'),
                    upsample_cfg=dict(type='InterpConv'),
                    norm_eval=False,
                    dcn=None,
                    plugins=None,
                    pretrained=None,
                    init_cfg=None):
            super(Unetdecode, self).__init__(init_cfg)

            self.pretrained = pretrained
            assert not (init_cfg and pretrained), \
                'init_cfg and pretrained cannot be setting at the same time'
            if isinstance(pretrained, str):
                warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                            'please use "init_cfg" instead')
                self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
            elif pretrained is None:
                if init_cfg is None:
                    self.init_cfg = [
                        dict(type='Kaiming', layer='Conv2d'),
                        dict(
                            type='Constant',
                            val=1,
                            layer=['_BatchNorm', 'GroupNorm'])
                    ]
            else:
                raise TypeError('pretrained must be a str or None')

            assert dcn is None, 'Not implemented yet.'
            assert plugins is None, 'Not implemented yet.'
            assert len(strides) == num_stages, \
                'The length of strides should be equal to num_stages, '\
                f'while the strides is {strides}, the length of '\
                f'strides is {len(strides)}, and the num_stages is '\
                f'{num_stages}.'
            assert len(enc_num_convs) == num_stages, \
                'The length of enc_num_convs should be equal to num_stages, '\
                f'while the enc_num_convs is {enc_num_convs}, the length of '\
                f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is '\
                f'{num_stages}.'
            assert len(dec_num_convs) == (num_stages-1), \
                'The length of dec_num_convs should be equal to (num_stages-1), '\
                f'while the dec_num_convs is {dec_num_convs}, the length of '\
                f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is '\
                f'{num_stages}.'
            assert len(downsamples) == (num_stages-1), \
                'The length of downsamples should be equal to (num_stages-1), '\
                f'while the downsamples is {downsamples}, the length of '\
                f'downsamples is {len(downsamples)}, and the num_stages is '\
                f'{num_stages}.'
            assert len(enc_dilations) == num_stages, \
                'The length of enc_dilations should be equal to num_stages, '\
                f'while the enc_dilations is {enc_dilations}, the length of '\
                f'enc_dilations is {len(enc_dilations)}, and the num_stages is '\
                f'{num_stages}.'
            assert len(dec_dilations) == (num_stages-1), \
                'The length of dec_dilations should be equal to (num_stages-1), '\
                f'while the dec_dilations is {dec_dilations}, the length of '\
                f'dec_dilations is {len(dec_dilations)}, and the num_stages is '\
                f'{num_stages}.'
            self.num_stages = num_stages
            self.strides = strides
            self.downsamples = downsamples
            self.norm_eval = norm_eval
            self.base_channels = base_channels

            self.encoder = nn.ModuleList()
            self.decoder = nn.ModuleList()

            for i in range(num_stages):
                enc_conv_block = []
                if i != 0:
                    if strides[i] == 1 and downsamples[i - 1]:
                        enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                    upsample = (strides[i] != 1 or downsamples[i - 1])
                    self.decoder.append(
                        UpConvBlock(
                            conv_block=BasicConvBlock,
                            in_channels=base_channels * 2**i,
                            skip_channels=base_channels * 2**(i - 1),
                            out_channels=base_channels * 2**(i - 1),
                            num_convs=dec_num_convs[i - 1],
                            stride=1,
                            dilation=dec_dilations[i - 1],
                            with_cp=with_cp,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            upsample_cfg=upsample_cfg if upsample else None,
                            dcn=None,
                            plugins=None))

                enc_conv_block.append(
                    BasicConvBlock(
                        in_channels=in_channels,
                        out_channels=base_channels * 2**i,
                        num_convs=enc_num_convs[i],
                        stride=strides[i],
                        dilation=enc_dilations[i],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        dcn=None,
                        plugins=None))
                self.encoder.append((nn.Sequential(*enc_conv_block)))
                in_channels = base_channels * 2**i

    def forward(self, x):
        self._check_input_divisible(x)
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        return dec_outs
    
    def forward_train(self, inputs, times, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs, times)
        
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses
    
    def forward_train_return_logits(self, inputs, times, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs, times)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses, seg_logits

    def forward_test(self, inputs, times, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, times)