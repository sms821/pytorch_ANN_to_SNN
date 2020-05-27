import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
from collections import OrderedDict
from . import spiking_activations
SpikeRelu = spiking_activations.SpikeRelu

class svhn_partial(nn.Module):

    def __init__(self, layer_num, layer_num_to_name, layer_name_to_type):
        super(svhn_partial, self).__init__()

        layer_list = []
        for l in range(layer_num, len(layer_num_to_name)-1):
            layer_name = layer_num_to_name[l]

            if type(layer_name_to_type[layer_name]) is not str:
                layer_list.append(layer_name_to_type[layer_name])
            #if layer_name in layer_name_to_type.keys():
            #    layer_list.append(layer_name_to_type[layer_name])

        self.layer_stack = nn.Sequential(*layer_list)
        last_layer_name = layer_num_to_name[len(layer_num_to_name)-1]
        self.fc = layer_name_to_type[last_layer_name]

    def forward(self, x):
        x = self.layer_stack(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SVHN(nn.Module):
    def __init__(self, thresholds, device, clp_slp, reset, features, n_channel, num_classes):
        super(SVHN, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes, bias=False)
        )
        self.relu = SpikeRelu(thresholds[11], 11, clp_slp, device, reset)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.relu(x)
        return x


def make_layers(thresholds, device, clp_slp, reset, cfg, batch_norm=False):
    layer_num = 0
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            layers += [SpikeRelu(thresholds[layer_num], layer_num, clp_slp, device, reset)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU(), nn.Dropout(0.3)]
            else:
                layers += [conv2d,
                           SpikeRelu(thresholds[layer_num], layer_num, clp_slp, device, reset),
                           nn.Dropout(0.3)]
            in_channels = out_channels
        layer_num += 1
    return nn.Sequential(*layers)


def svhn_spike(thresholds, device, clamp_slope, reset, n_channel=32, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']

    layers = make_layers(thresholds, device, clamp_slope, reset, cfg, batch_norm=False)

    model = SVHN(thresholds, device, clamp_slope, reset, layers, n_channel=8*n_channel, num_classes=10)
    return model
