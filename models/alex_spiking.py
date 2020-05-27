import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from . import spiking_activations
SpikeRelu = spiking_activations.SpikeRelu

__all__ = ['AlexNet', 'alexnet']


class alexnet_partial(nn.Module):

    def __init__(self, layer_num, layer_num_to_name, layer_name_to_type):
        super(alexnet_partial, self).__init__()

        layer_list1 = []
        layer_list2 = []
        layer_list3 = []
        if layer_num <= 16:
            self.flat = 256 * 6 * 6
            for l in range(layer_num, 16):
                #print(layer_num_to_name[l])
                layer_type = layer_name_to_type[layer_num_to_name[l]]
                if type(layer_type) is not str:
                    layer_list1.append(layer_type)
            for l in range(16, len(layer_num_to_name)):
                layer_type = layer_name_to_type[layer_num_to_name[l]]
                if type(layer_type) is not str:
                    layer_list2.append(layer_type)
        else:
            self.flat = 4096
            for l in range(layer_num, len(layer_num_to_name)):
                layer_type = layer_name_to_type[layer_num_to_name[l]]
                layer_list2.append(layer_type)

        self.layer_stack1 = nn.Sequential(*layer_list1)
        self.layer_stack2 = nn.Sequential(*layer_list2)

    def forward(self, x):
        x = self.layer_stack1(x)
        x = x.view(-1, self.flat)
        x = self.layer_stack2(x)
        return x


class BinConv2d(nn.Module): # change the name of BinConv2d
    def __init__(self, input_channels, output_channels,
            vth, th_idx, clp_slp, device, reset,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0.0,
            Linear=False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        else:
            self.linear = nn.Linear(input_channels, output_channels, bias=False)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = SpikeRelu(vth, th_idx, clp_slp, device, reset)

    def forward(self, x):
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x



class AlexNet_spiking(nn.Module):
    def __init__(self, thresholds, clp_slp, device, reset, num_classes=1000):
        super(AlexNet_spiking, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
            SpikeRelu(thresholds[0], 0, clp_slp, device, reset),

            nn.AvgPool2d(kernel_size=3, stride=2),
            SpikeRelu(thresholds[1], 1, clp_slp, device, reset),

            BinConv2d(96, 256, thresholds[2], 2, clp_slp, device, reset, kernel_size=5, stride=1, padding=2, groups=1),

            nn.AvgPool2d(kernel_size=3, stride=2),
            SpikeRelu(thresholds[3], 3, clp_slp, device, reset),

            BinConv2d(256, 384, thresholds[4], 4, clp_slp, device, reset,  kernel_size=3, stride=1, padding=1),
            BinConv2d(384, 384, thresholds[5], 5, clp_slp, device, reset,  kernel_size=3, stride=1, padding=1, groups=1, dropout=.1),
            BinConv2d(384, 256, thresholds[6], 6, clp_slp, device, reset,  kernel_size=3, stride=1, padding=1, groups=1, dropout=.1),

            nn.AvgPool2d(kernel_size=3, stride=2),
            SpikeRelu(thresholds[7], 7, clp_slp, device, reset),
        )
        self.classifier = nn.Sequential(
            BinConv2d(256 * 6 * 6, 4096, thresholds[8], 8, clp_slp, device, reset,  Linear=True),
            BinConv2d(4096, 4096, thresholds[9], 9, clp_slp, device, reset,  Linear=True, dropout=0.1),

            nn.Linear(4096, num_classes, bias=False),
            SpikeRelu(thresholds[10], 10, clp_slp, device, reset),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet_spiking(thresholds, device, clp_slp, reset, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet_spiking(thresholds, clp_slp, device, reset, **kwargs)
    #print(model.state_dict().keys())
    #model.features = torch.nn.DataParallel(model.features)
    return model

