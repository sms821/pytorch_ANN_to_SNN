import os
import errno
import sys
import time
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

import ast
import json

from common import *

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def compute_thresholds(net, dataloader, out_dir, percentile=99.9, device='cuda:0', spiking=True, find_zeros=False):
    relus = []
    relu_names = []
    ftr_zeros_dict = {}
    for k,v in net.named_modules():
        if isinstance( v, nn.Conv2d) or \
            isinstance(v, nn.Linear) or \
            isinstance(v, nn.AdaptiveAvgPool2d) or \
            isinstance(v, nn.AvgPool2d):
            relus.append(v)
            relu_names.append(k)
            ftr_zeros_dict[k] = 0

    hooks = [Hook(layer) for layer in relus]
    print('number of spike layers with thresholds: {}'.format(len(hooks)))

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    acts = np.zeros((len(hooks)+1, 50000))
    if find_zeros:
        ftr_zeros = torch.zeros((len(hooks), 10000))
        #ftr_zeros = torch.zeros(len(hooks))
        prev_batch_size = 0
    with torch.no_grad():
        for n, data in enumerate(dataloader):
            images, targets = data
            images, targets = images.to(device), targets.to(device)

            outputs = net(images)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            batch_idx = n
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            batch_size = targets.size(0)
            #print(batch_size)

            img_max = np.amax(images.cpu().numpy(), axis=(1,2,3))
            acts[0,n*batch_size:(n+1)*batch_size] = img_max

            for i, hook in enumerate(hooks):
                batch_mean = ftr_zeros_dict[relu_names[i]]

                #print(hook.output.size())
                if len(hook.output.size()) > 2:
                    acts[i+1][n*batch_size:(n+1)*batch_size] = np.amax(hook.output.cpu().numpy(), axis=(1,2,3))

                    if find_zeros:
                        layer_sz = hook.output.size()
                        layer_sz = layer_sz[1] * layer_sz[2] * layer_sz[3]
                        #print(layer_sz)
                        batch_zeros = (1/layer_sz)*torch.sum(torch.where(hook.output > 0, torch.ones(1, device=device), torch.zeros(1, device=device)), dim=(1,2,3))
                        ftr_zeros[i][n*batch_size:(n+1)*batch_size] = batch_zeros

                else:
                    acts[i+1][n*batch_size:(n+1)*batch_size] = np.amax(hook.output.cpu().numpy(), axis=1)

                    if find_zeros:
                        layer_sz = hook.output.size()
                        layer_sz = layer_sz[1]
                        batch_zeros = (1/layer_sz)*torch.sum(torch.where(hook.output > 0, torch.ones(1, device=device), torch.zeros(1, device=device)), dim=1)
                        ftr_zeros[i][n*batch_size:(n+1)*batch_size] = batch_zeros

                prev_batch_size = batch_size


    max_val = np.percentile(acts, percentile, axis=1)
    print('{}th percentile of max activations: {}'.format(percentile, max_val))

    if spiking:
        thresholds = torch.zeros(len(max_val)-1)
        for i in range(len(thresholds)):
            thresholds[i] = max_val[i+1] / max_val[i]
        np.savetxt(os.path.join(out_dir, 'thresholds.txt'), thresholds, fmt='%.5f')
        print('thresholds: ', thresholds)
        filenm = 'max_acts.txt'

    elif find_zeros:
        for r in range(len(hooks)):
            ftr_zeros_dict[relu_names[r]] = torch.mean(ftr_zeros[r])

        for k,v in ftr_zeros_dict.items():
            print(k, v)
            ftr_zeros_dict[k] = v.cpu().data.numpy().tolist()

        filenm = 'avg_sparsity.json'
        j_str = json.dumps(ftr_zeros_dict, indent=2)
        print(ftr_zeros_dict)
        print(j_str)
        with open(os.path.join(out_dir, filenm), "w") as f:
            f.write(j_str)
            f.close()
    else:
        filenm = 'max_acts_{}.txt'.format(percentile)

    np.savetxt(os.path.join(out_dir, filenm), max_val, fmt='%.5f')


def copy_weights(new_net, net):
    " true copies weights from net to new_net "
    layer_num = 0
    net_dict = {}
    for _,t in net.named_modules():
        if isinstance(t, nn.Conv2d) or isinstance(t, nn.Linear):
            net_dict[layer_num] = t
            layer_num += 1

    layer_num = 0
    new_net_dict = {}
    for _,t in new_net.named_modules():
        if isinstance(t, nn.Conv2d) or isinstance(t, nn.Linear):
            new_net_dict[layer_num] = t
            layer_num += 1

    for n,t in net_dict.items():
        new_layer = new_net_dict[n]
        new_layer.weight.data = t.weight.data.clone()

        if t.bias is not None:
            new_layer.bias.data = t.bias.data.clone()


def adjust_weights(wt_layer, bn_layer):
    num_out_channels = wt_layer.weight.size()[0]

    bias = torch.zeros(num_out_channels)
    wt_layer_bias = torch.zeros(num_out_channels)
    if wt_layer.bias is not None:
        wt_layer_bias = wt_layer.bias

    wt_cap = torch.zeros(wt_layer.weight.size())
    for i in range(num_out_channels):
        beta, gamma = 0, 1

        if bn_layer.weight is not None:
            gamma = bn_layer.weight[i]
        if bn_layer.bias is not None:
            beta = bn_layer.bias[i]

        sigma = bn_layer.running_var[i]
        mu = bn_layer.running_mean[i]
        eps = bn_layer.eps
        scale_fac = gamma / torch.sqrt(eps+sigma)
        wt_cap[i,:,:,:] = wt_layer.weight[i,:,:,:]*scale_fac
        bias[i] = (wt_layer_bias[i]-mu)*scale_fac + beta
    return (wt_cap, bias)


def merge_bn(model, model_nobn):

    "merges bn params with those of the previous layer"
    "works for the layer pattern: conv->bn only"

    # Serialize the original model
    name_to_type = serialize_model(model)
    key_list = list(name_to_type.keys())

    # Serialize the nobn model
    name_to_type_nobn = serialize_model(model_nobn)
    conv_names = []
    for k,v in name_to_type_nobn.items():
        if type(v) == nn.Conv2d or type(v) == nn.Linear:
            conv_names.append(k)

    nobn_num = 0
    layer_num = 0
    for i,n in enumerate(key_list):
        if isinstance(name_to_type[n], nn.Conv2d) and \
                isinstance(name_to_type[key_list[i+1]], nn.BatchNorm2d):

            conv_layer = name_to_type[n]
            bn_layer = name_to_type[key_list[i+1]]
            new_wts, new_bias = adjust_weights(conv_layer, bn_layer)

            nobn_name = conv_names[layer_num]
            conv_layer_nobn = name_to_type_nobn[nobn_name]

            conv_layer_nobn.weight.data = new_wts
            if conv_layer_nobn.bias is not None:
                conv_layer_nobn.bias.data = new_bias

            layer_num += 1

        elif isinstance(name_to_type[n], nn.Conv2d) or \
                isinstance(name_to_type[n], nn.Linear):
            layer = name_to_type[n]

            nobn_name = conv_names[layer_num]

            layer_nobn = name_to_type_nobn[nobn_name]
            layer_nobn.weight.data = layer.weight.data.clone()
            if layer.bias is not None and layer_nobn.bias is not None:
                layer_nobn.bias.data = layer.bias.data.clone()

            layer_num += 1

    return model_nobn


def has_bn(net):
    for m in net.modules():
        if type(m) == nn.BatchNorm2d:
            return True
    return False


import numpy as np
import matplotlib.pyplot as plt
def validate(net, testloader, device='cuda:0'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    display_imgs = False
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #print(inputs.size())
            if display_imgs:
                inp = inputs[0].numpy().transpose((1, 2, 0))
                inp = np.clip(inp, 0, 1)
                plt.imshow(inp)
                plt.show()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), acc, correct, total))
    return acc


def load_imagenet(data_dir, batch_size=128, shuffle=True):
    """
    Load the ImageNet dataset.
    """
    #print('loading tiny imagenet')
    train_dir = os.path.join(data_dir, 'Train_Data')
    test_dir = os.path.join(data_dir, 'Validation_Data')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]) # using mean and std of original imagenet dataset

    #print('reading data..')

    train_transform = transforms.Compose([
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    train_loader = None


    test_transform = transforms.Compose([
        transforms.Resize(256), # this line is imp for pre-trained imagenet models to yield reported acc.
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = datasets.ImageFolder(test_dir, test_transform)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return (train_loader, val_loader)


def load_cifar10(data_dir='./data', arch='mobilenet_cifar10', batch_size=128, class_num=-1):
    #print('class_num: {}'.format(class_num))
    # Data
    print('==> Preparing data..')
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if arch == 'vgg_cifar10':
        std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_test)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    targets = torch.tensor(testset.targets)
    target_idx, sampler = None, None
    if class_num >= 0:
        target_idx = (targets==class_num).nonzero()
        sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx)

    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return (trainloader, testloader)


def load_cifar100(data_dir='./data', arch='mobilenet_cifar10', batch_size=128, class_num=-1):
    MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_test)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    targets = torch.tensor(testset.targets)
    target_idx, sampler = None, None
    if class_num >= 0:
        target_idx = (targets==class_num).nonzero()
        sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx)

    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=2)
    return (trainloader, testloader)



def load_svhn(data_dir='./data', arch='mobilenet_cifar10', batch_size=128, class_num=-1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    trainset = datasets.SVHN( root=data_dir, split='train', download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root=data_dir, split='test', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            #target_transform=target_transform
        ),
        batch_size=batch_size, shuffle=False )
    return (trainloader, testloader)


def load_mnist(data_dir='./data', arch='mobilenet_cifar10', batch_size=128, class_num=-1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081]) ])

    trainset = datasets.MNIST(root=data_dir, train=True, transform=transform)
    testset = datasets.MNIST(root=data_dir, train=False, transform=transform)

    trainloader = data.DataLoader( testset, batch_size=batch_size, shuffle=False, num_workers=4 )
    testloader = data.DataLoader( testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader


def save_model(net, state, model_path, file_name):
    assert os.path.isdir(model_path), 'Error: no {} directory found!'.format(model_path)
    file_path = os.path.join(model_path, file_name)
    print('Saving..')
    state['net'] = net.state_dict()
    torch.save(state, file_path)


def load_model(net, model_path, file_name):
    # Load checkpoint.
    file_path = os.path.join(model_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print (best_acc, start_epoch)
    return checkpoint, net


def read_max_acts(out_dir):
    max_acts = []
    with open(os.path.join(out_dir, 'max_acts.txt')) as f:
        for line in f.readlines():
            a,b = line.rstrip('\n').split(',')
            max_acts.append((float(a), b))
    return max_acts


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
term_width = '143'
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
