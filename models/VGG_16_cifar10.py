import os
import numpy as np

import torch
import torch.nn as nn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


VGG_16_cifar10 = nn.Sequential( # Sequential,
	nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.AvgPool2d((2, 2),(2, 2)),
	nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.AvgPool2d((2, 2),(2, 2)),
	nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
	nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
	Lambda(lambda x: x.view(x.size(0),-1)), # View,
	nn.Sequential( # Sequential,
		nn.Dropout(0.1),
		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,512,bias=False)), # Linear,
		nn.ReLU(),
		nn.Dropout(0.1),
		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,10,bias=False)), # Linear,
	),
)

