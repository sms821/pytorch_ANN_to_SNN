import torch
import torch.nn as nn
from . import spiking_activations
SpikeRelu = spiking_activations.SpikeRelu

################ Spiking Mobilenet: mobnet with relu layers replaced by Spike Relus################

class Block_spike(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, thresholds, device, th_idx=0, clp_slp=0, stride=1, reset='to-threshold'):
        super(Block_spike, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=True)
        self.relu1 = SpikeRelu(thresholds[th_idx], th_idx, clp_slp, device, reset)

        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu2 = SpikeRelu(thresholds[th_idx+1], th_idx+1, clp_slp, device, reset)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        return out


def mobilenet_mod_spike(thresholds, device, clp_slp=0, num_classes=10, reset='to=threshold'):
    class MobileNet_mod_spike(nn.Module):
        # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
        cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

        def __init__(self, thresholds,  device, clp_slp=0, num_classes=10, reset='to-threshold'):
            super(MobileNet_mod_spike, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
            self.relu1 = SpikeRelu(thresholds[0], 0, clp_slp, device, reset)

            self.layers = self._make_layers(thresholds, clp_slp, device, reset, in_planes=32 )
            idx = 2*len(self.layers)

            self.avgpool = nn.AvgPool2d(2)
            self.relu_avg = SpikeRelu(thresholds[-2], idx+1, clp_slp, device, reset)
            self.linear = nn.Linear(1024, num_classes, bias=False)
            self.relu_lin = SpikeRelu(thresholds[-1], idx+2, clp_slp, device, reset)

        def _make_layers(self, thresholds, clp_slp, device, reset, in_planes ):
            layers = []
            idx = 1
            for x in self.cfg:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(Block_spike(in_planes, out_planes, thresholds, device, 2*idx-1, clp_slp, stride, reset))
                in_planes = out_planes
                idx += 1
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.relu1(self.conv1(x))
            out = self.layers(out)
            out = self.avgpool(out)
            out = self.relu_avg(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            out = self.relu_lin(out)
            return out

    return  MobileNet_mod_spike(thresholds, device, clp_slp, num_classes, reset)
