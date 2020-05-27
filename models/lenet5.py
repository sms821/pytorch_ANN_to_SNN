import torch
import torch.nn as nn
from . import spiking_activations
SpikeRelu = spiking_activations.SpikeRelu

class lenet5(nn.Module):

    def __init__(self):
        super(lenet5, self).__init__()

        # set-1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2, bias=False)
        self.bin1 = nn.ReLU()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # set-2
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2, bias=False)
        self.bin2 = nn.ReLU()
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # set-3
        self.fc3 = nn.Linear(50*7*7, 500, bias=False)
        self.bin3 = nn.ReLU()

        # set-4
        self.fc4 = nn.Linear(500, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bin1(x)

        x = self.avg_pool1(x)

        x = self.conv2(x)
        x = self.bin2(x)

        x = self.avg_pool2(x)

        x = x.view(-1, 7*7*50)
        x = self.fc3(x)
        x = self.bin3(x)

        x = self.fc4(x)

        return x

def lenet5_spiking(thresholds, device, clamp_slope, reset):
    class Lenet5_spike(nn.Module):
        def __init__(self, thresholds, device, clp_slp=0, reset='to-threshold'):
            super(Lenet5_spike, self).__init__()

            # set-1
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2, bias=False)
            self.bin1 = SpikeRelu(thresholds[0], 0, clp_slp, device, reset)
            self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.avg_spike1 = SpikeRelu(thresholds[1], 1, clp_slp, device, reset)

            # set-2
            self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2, bias=False)
            self.bin2 = SpikeRelu(thresholds[2], 2, clp_slp, device, reset)
            self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.avg_spike2 = SpikeRelu(thresholds[3], 3, clp_slp, device, reset)

            # set-3
            self.fc3 = nn.Linear(50*7*7, 500, bias=False)
            self.bin3 = SpikeRelu(thresholds[4], 4, clp_slp, device, reset)

            # set-4
            self.fc4 = nn.Linear(500, 10, bias=False)
            self.bin4 = SpikeRelu(thresholds[5], 5, clp_slp, device, reset)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bin1(x)

            x = self.avg_pool1(x)
            x = self.avg_spike1(x)

            x = self.conv2(x)
            x = self.bin2(x)

            x = self.avg_pool2(x)
            x = self.avg_spike2(x)

            x = x.view(-1, 7*7*50)
            x = self.fc3(x)
            x = self.bin3(x)

            x = self.fc4(x)
            x = self.bin4(x)
            return x
    return Lenet5_spike(thresholds, device, clamp_slope, reset)

