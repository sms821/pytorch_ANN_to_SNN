import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np

from models.spiking_activations import spikeRelu


def serialize_model(model):
    "gives relative ordering of layers in a model:"
    "layer-name => layer-type"

    name_to_type = OrderedDict()
    layer_num = 0
    for name, module in model.named_modules():
        #print(name)
        if isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ReLU) or \
                isinstance(module, nn.Linear) or \
                isinstance(module, nn.AvgPool2d) or \
                isinstance(module, nn.BatchNorm2d) or \
                isinstance(module, spikeRelu) or \
                isinstance(module, nn.ReLU6) or \
                isinstance(module, nn.AdaptiveAvgPool2d):

            name_to_type[name] = module
            layer_num += 1

    return name_to_type

class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()
