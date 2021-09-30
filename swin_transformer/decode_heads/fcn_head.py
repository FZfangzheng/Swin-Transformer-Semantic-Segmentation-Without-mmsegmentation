import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


from .decode_head import BaseDecodeHead
from ..losses.accuracy import accuracy
from mmseg.ops import resize

class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self, in_channels=512, in_index=2, channels=256, num_convs=1, concat_input=False, dropout_ratio=0.1, num_classes=3,
                 norm_cfg=dict(type='BN', requires_grad=True), align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4), kernel_size=3):
        super(FCNHead, self).__init__()
        assert num_convs >= 0
        self.num_convs = num_convs
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        # self.criteria = nn.CrossEntropyLoss()
        self.norm_cfg = norm_cfg

        self.in_index = in_index
        # self.loss_decode = build_loss(loss_decode)

        self.align_corners = align_corners


        if num_convs == 0:
            assert self.in_channels == self.channels

        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output


    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = 0.4*torch.nn.functional.cross_entropy(seg_logit, seg_label, ignore_index=self.ignore_index)
        # loss['loss_seg'] = 0.4*self.criteria(seg_logit, seg_label, ignore_index=self.ignore_index)
        # loss['loss_seg'] = 0.4 * self.loss_decode(
        #     seg_logit,
        #     seg_label,
        #     ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
