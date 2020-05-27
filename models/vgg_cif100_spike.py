import torch
import torch.nn as nn
from . import spiking_activations
SpikeRelu = spiking_activations.SpikeRelu

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

#######################################################################################
############################## nobn version of vgg-net ################################
class VGG_nobn_spike(nn.Module):

    def __init__(self, thresholds, device, clamp_slope, reset, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096, bias=False),
            SpikeRelu(thresholds[15], 15, clamp_slope, device, reset),
            #nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096, bias=False),
            SpikeRelu(thresholds[16], 16, clamp_slope, device, reset),
            #nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, num_class, bias=False),
            SpikeRelu(thresholds[17], 17, clamp_slope, device, reset)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers_spike(thresholds, device, clamp_slope, reset, cfg, batch_norm=False):
    layers = []

    layer_num = 0
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            layers += [SpikeRelu(thresholds[layer_num], layer_num, clamp_slope, device, reset)]
            layer_num += 1
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1, bias=True)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        #layers += [nn.ReLU(inplace=True)]
        layers += [SpikeRelu(thresholds[layer_num], layer_num, clamp_slope, device, reset)]

        input_channel = l
        layer_num += 1

    return nn.Sequential(*layers)


def vgg13_nobn_spike(thresholds, device, clamp_slope, reset):
    return VGG_nobn_spike(thresholds, device, clamp_slope, reset, make_layers_spike(thresholds, device, clamp_slope, reset, cfg['B'], batch_norm=False))

