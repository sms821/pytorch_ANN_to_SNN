import torch
import torch.nn as nn
from . import spiking_activations
SpikeRelu = spiking_activations.SpikeRelu


class DepthSeperabelConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, thresholds, device, th_idx=0, clp_slp=0, reset='to-threshold', **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            SpikeRelu(thresholds[th_idx], th_idx, clp_slp, device, reset)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=True),
            SpikeRelu(thresholds[th_idx+1], th_idx+1, clp_slp, device, reset)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, thresholds, device, clp_slp, reset, **kwargs):

        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.relu = SpikeRelu(thresholds[0], 0, clp_slp, device, reset)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x


class MobileNet(nn.Module):

    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """

    def __init__(self, thresholds, device, clp_slp, reset, width_multiplier=1, class_num=100):

       super().__init__()

       alpha = width_multiplier
       self.stem = nn.Sequential(
           BasicConv2d(3, int(32 * alpha), 3, thresholds, device, clp_slp, reset, padding=1, bias=True),

           DepthSeperabelConv2d(
               int(32 * alpha),
               int(64 * alpha),
               3,
               thresholds, device, 1, clp_slp, reset,
               padding=1,
               bias=True
           )
       )

       #downsample
       self.conv1 = nn.Sequential(
           DepthSeperabelConv2d(
               int(64 * alpha),
               int(128 * alpha),
               3,
               thresholds, device, 3, clp_slp, reset,
               stride=2,
               padding=1,
               bias=True
           ),
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(128 * alpha),
               3,
               thresholds, device, 5, clp_slp, reset,
               padding=1,
               bias=True
           )
       )

       #downsample
       self.conv2 = nn.Sequential(
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(256 * alpha),
               3,
               thresholds, device, 7, clp_slp, reset,
               stride=2,
               padding=1,
               bias=True
           ),
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(256 * alpha),
               3,
               thresholds, device, 9, clp_slp, reset,
               padding=1,
               bias=True
           )
       )

       #downsample
       self.conv3 = nn.Sequential(
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(512 * alpha),
               3,
               thresholds, device, 11, clp_slp, reset,
               stride=2,
               padding=1,
               bias=True
           ),

           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               thresholds, device, 13, clp_slp, reset,
               padding=1,
               bias=True
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               thresholds, device, 15, clp_slp, reset,
               padding=1,
               bias=True
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               thresholds, device, 17, clp_slp, reset,
               padding=1,
               bias=True
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               thresholds, device, 19, clp_slp, reset,
               padding=1,
               bias=True
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               thresholds, device, 21, clp_slp, reset,
               padding=1,
               bias=True
           )
       )

       #downsample
       self.conv4 = nn.Sequential(
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(1024 * alpha),
               3,
               thresholds, device, 23, clp_slp, reset,
               stride=2,
               padding=1,
               bias=True
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               thresholds, device, 25, clp_slp, reset,
               padding=1,
               bias=True
           )
       )

       self.avg = nn.AdaptiveAvgPool2d(1)
       self.relu_avg = SpikeRelu(thresholds[27], 27, clp_slp, device, reset)

       self.fc = nn.Linear(int(1024 * alpha), class_num, bias=False)
       self.relu_lin = SpikeRelu(thresholds[28], 28, clp_slp, device, reset)

    def forward(self, x):
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = self.relu_avg(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.relu_lin(x)
        return x


def mobilenet_cif100_nobn_spike(thresholds, device, clp_slp=0, reset='to=threshold', alpha=1, class_num=100):
    return MobileNet(thresholds, device, clp_slp, reset, alpha, class_num)

