import numpy as np

import torch
import torch.nn as nn

from utils import *
from common import *
import math

import json
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd

from models.spiking_activations import SpikeRelu, spikeRelu

################# Spiking VGG-net: uses spike relus #################################

def simulate_spike_model(net, arch, val_loader, config, thresholds, max_acts, \
        num_classes, img_size, sbi, model_partial, device='cuda:0'):

    out_dir = config['DEFAULT']['app_name']
    spike_config = config['spiking']
    batch_size = config.getint('original model','batch_size')
    class_num = config.getint('original model','class')
    clamp_slope = spike_config.getfloat('clamp_slope')
    time_window = spike_config.getint('time_window')
    numBatches = spike_config.getint('num_batches')
    hybrid = spike_config.getboolean('hybrid_model')
    split_layer = spike_config.getint('split_layer')
    save_activation = spike_config.getboolean('save_activations')
    save_correlation = spike_config.getboolean('save_correlation')
    write_to_file = spike_config.getboolean('write_to_file')
    MFR = spike_config.getfloat('MFR')

    net.eval()

    spike_net = createSpikingModel(net, arch, num_classes, spike_config, \
            thresholds, max_acts, device, out_dir)
    print(spike_net)
    spike_net.eval()
    if spike_config.getboolean('plot_mean_var'):
        plot_mean_var(net, spike_net, out_dir)

    buffers = None
    hooks = None
    num_layers = None
    layers = []
    if save_activation or save_correlation:
        hooks, buffers = create_buffers(net, img_size, device, batch_size)
        num_layers = len(buffers)
        image_corr = np.zeros(numBatches*batch_size)
        layer_corr = np.zeros((num_layers, numBatches*batch_size))


    total_correct = 0
    expected_correct = 0
    combined_model_correct = 0
    total_images = 0
    batch_num = 0
    confusion_matrix = np.zeros([num_classes,num_classes], int)
    out_spikes_t_b_c = torch.zeros((time_window, batch_size, num_classes))
    confidence_s = np.zeros(numBatches*batch_size)
    confidence_a = np.zeros(numBatches*batch_size)
    spike_buffers = None

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if batch_num >= numBatches:
                break
            print ('\n\n------------ inferring batch {} -------------'.format(batch_num))
            images, labels = data

            # perform inference on the original model to compare
            images = images.to(device)
            net = net.to(device)
            output_org = net(images.float())
            _, predicted_org = torch.max(output_org.data, 1)

            # create the spiking model
            spike_net = createSpikingModel(net, arch, num_classes, spike_config, \
                    thresholds, max_acts, device, out_dir)
            #sanity_check(net, spike_net, max_acts)
            spike_net = spike_net.to(device)
            spike_net.eval()

            #### record outputs of intermediate layers of original model
            spike_hooks = None
            if save_activation or save_correlation:

                buffers[0] = images
                for z,h in enumerate(hooks):
                    layer_out = h.output
                    buffers[z+1] = layer_out

                spike_hooks, spike_buffers = create_spike_buffers(spike_net, img_size, device, batch_size)
                assert(len(buffers) == len(spike_buffers))
                for i in range(len(buffers)):
                    assert(buffers[i].size() == spike_buffers[i].size())

                assert(len(spike_hooks) == len(hooks) == num_layers-1)

            # starting of time-stepped spike integration of SNN
            for t in range(time_window):

                # convert image pixels to spiking inputs
                spikes = poisson_spikes(images.cpu().numpy(), MFR)

                # supply random inputs
                spikes = torch.from_numpy(spikes)

                out_spikes = None
                if spike_config.get('poisson_spikes'):

                    # supplying spiking inputs to the spiking model
                    spikes = spikes.to(device)
                    out_spikes = spike_net(spikes.float())
                else:

                    # supplying original analog inputs to the spiking model
                    images = images.to(device)
                    out_spikes = spike_net(images.float())

                out_spikes_t_b_c[t,:,:] = out_spikes


                # record sum of output spikes of intermediate layers of spiking model
                if save_activation or save_correlation:
                    for z in range(len(spike_hooks)+1):
                        for b in range(batch_size):
                            if z == 0:
                                spike_buffers[0][b,:] += spikes[b].float()
                            else:
                                h = spike_hooks[z-1]
                                spike_buffers[z][b,:] += h.output[b]

            # end of time-stepped spike integration

            if save_activation or save_correlation:
                # corresponding analog value of these spikes
                for z in range(len(spike_hooks)+1):
                    spike_buffers[z] = (spike_buffers[z] / time_window) * max_acts[z]

            ############### calling the partial non-spiking model here ##############

            predicted_partial = -1
            if hybrid:
                assert save_activation, 'set `save_activations` to True in config-file.'
                sbi, model_partial = create_partial_model(split_layer, net, spike_net, arch)
                #print(model_partial)
                # sbi: spike buffer index
                temp = spike_buffers[sbi+1].float()

                model_partial = model_partial.to(device)
                output_partial = model_partial(temp.to(device))
                #print(output_partial)

                _, predicted_partial = torch.max(output_partial.data, 1)
                combined_model_correct += (predicted_partial.cpu() == labels).sum().item()

            ############### calling the partial non-spiking model here (ENDS)##############

            if save_activation or save_correlation:
                # save the correlation coefficients between spike acts and analog acts
                for l in range(len(hooks)+1):
                    if len(buffers[l].size()) > 2:
                        B,C,H,W = buffers[l].size()
                        buffers[l] = buffers[l].view(-1, C*H*W)
                        spike_buffers[l] = spike_buffers[l].view(-1, C*H*W)
                    else:
                        B,C = buffers[l].size()
                        buffers[l] = buffers[l].view(-1, C)
                        spike_buffers[l] = spike_buffers[l].view(-1, C)

                #print('layer_corr: {}'.format(layer_corr.shape))
                for l in range(len(hooks)):
                    i = 0
                    for b in range(batch_size):
                        corr_layer = np.corrcoef(buffers[l][b].cpu(), spike_buffers[l][b].cpu())
                        layer_corr[l][batch_num*batch_size+b] = corr_layer[0,1]
                        i = i+1


            # accumulating output spikes for all images in a batch
            total_spikes_b_c = torch.zeros((batch_size, num_classes))
            for b in range(batch_size):
                total_spikes_per_input = torch.zeros((num_classes))
                for t in range(time_window):
                    total_spikes_per_input += out_spikes_t_b_c[t,b,:]
                #print ("total spikes per output: {}".format(total_spikes_per_input ))
                total_spikes_b_c[b,:] = total_spikes_per_input
                #total_spikes_b_c[b,:] = total_spikes_per_input / time_window # note the change

            _, predicted = torch.max(total_spikes_b_c.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            expected_correct += (predicted_org.cpu() == labels).sum().item()

            print('snn: {}\tann: {}\tpart: {}\tlabel: {}'.\
                    format(predicted, predicted_org, predicted_partial, labels))

            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

            batch_num += 1

            model_accuracy = total_correct / total_images * 100
            expected_acc = expected_correct / total_images * 100
            print('Model accuracy on {} test images: {}%\t\t\tacc. of ANN: {}%'.format(total_images, model_accuracy, expected_acc))
            if hybrid:
                combined_model_acc = combined_model_correct / total_images * 100
                print ('Combined model accuracy: {}%'.format(combined_model_acc))

    model_accuracy = total_correct / total_images * 100
    expected_acc = expected_correct / total_images * 100
    print('Model accuracy on {} test images: {}%\t\t\tacc. of ANN: {}%'.format(total_images, model_accuracy, expected_acc))
    if hybrid:
        combined_model_acc = combined_model_correct / total_images * 100
        print ('Combined model accuracy: {}%'.format(combined_model_acc))

    if class_num < 0:
        class_num = ''
    if write_to_file and save_correlation:
        print('[INFO] Saving correlations...')
        np.save(os.path.join(out_dir, 'layerwise_corr'+str(class_num)), layer_corr)

    if write_to_file and save_activation:
        print('[INFO] Saving activations...')
        for s in range(len(spike_buffers)):
            spike_buffers[s] = spike_buffers[s].cpu().numpy()
        np.savez(os.path.join(out_dir, 'layerwise_acts'+str(class_num)), *spike_buffers)


def create_spike_buffers(net, img_size, device='cuda:0', B=1):
    relus = []
    for m in net.modules():
        if isinstance(m, spikeRelu):
            relus.append(m)

    hooks = [Hook(layer) for layer in relus]
    mats = create_mats(net, img_size, hooks, device, B)

    return hooks, mats


def create_buffers(net, img_size, device='cuda:0', B=1):
    name_to_type = serialize_model(net)
    key_list = list(name_to_type.keys())
    relus = []
    for i in range(len(key_list)):
        if i < len(key_list)-1 and \
            isinstance(name_to_type[key_list[i]], nn.Conv2d) and \
                (isinstance(name_to_type[key_list[i+1]], nn.ReLU) or \
                isinstance(name_to_type[key_list[i+1]], nn.ReLU6)):
            relus.append(name_to_type[key_list[i+1]])

        elif i < len(key_list)-1 and \
            isinstance(name_to_type[key_list[i]], nn.Linear) and \
                (isinstance(name_to_type[key_list[i+1]],nn.ReLU) or \
                isinstance(name_to_type[key_list[i+1]], nn.ReLU6)):
            relus.append(name_to_type[key_list[i+1]])

        elif isinstance(name_to_type[key_list[i]], nn.Linear) or \
                isinstance(name_to_type[key_list[i]], nn.AvgPool2d) or \
                isinstance(name_to_type[key_list[i]], nn.AdaptiveAvgPool2d):
            relus.append(name_to_type[key_list[i]])

    hooks = [Hook(layer) for layer in relus]
    mats = create_mats(net, img_size, hooks, device, B)

    return hooks, mats


def create_mats(net, img_size, hooks, device='cuda:0', B=1):
    inp_size = [B]
    inp_size = inp_size + list(img_size[1:])

    inp = torch.zeros(inp_size).to(device)
    outputs = net(inp.float())

    mats = []
    mats.append(torch.zeros(inp_size).to(device))
    for h in hooks:
        shape = h.output.size()
        if len(shape) > 2:
            curr_shape = [B, shape[1], shape[2], shape[3]]
        else:
            curr_shape = [B, shape[1]]
        temp = torch.zeros(curr_shape).to(device)
        mats.append(temp)

    return mats


def print_dict(dict_):
    for i, (k,v) in enumerate(dict_.items()):
        print(i, k, v)

def create_partial_model(split_layer_num, model, spike_model, arch):
    " returns the last few layers from `split_layer_num` onwards "

    n_to_t_ann = dict(serialize_model(model))

    num_to_name_ann = {}
    i = 0
    for k,v in n_to_t_ann.items():
        num_to_name_ann[i] = k
        if isinstance(v, nn.AvgPool2d):
            i = i+1
            num_to_name_ann[i] = 'spike{}'.format(i)
        i = i+1

    n_to_t_snn = serialize_model(spike_model)

    num_to_name_snn = {}
    spike_index = {}
    i = 0
    s = 0
    for k,v in n_to_t_snn.items():
        num_to_name_snn[i] = k
        if isinstance(v, spikeRelu):
            spike_index[i] = s
            s += 1
        i = i+1
    assert(len(num_to_name_ann) == len(num_to_name_snn)-1)

    # adjust n_to_t_ann to have same # elements as n_to_snn
    for k,v in num_to_name_ann.items():
        if 'spike' in v:
            n_to_t_ann[v] = v

    model_partial = None
    si = None # spike buffer index
    if type(n_to_t_snn[num_to_name_snn[split_layer_num]]) == type(n_to_t_ann[num_to_name_ann[split_layer_num]]): # non-spike layer

        # send num_to_name_ann and layer num to partial model class
        SPLIT_LAYER = split_layer_num
        # find index of the earliest spike layer
        for s in range(split_layer_num-1, 0, -1):
            if isinstance(n_to_t_snn[num_to_name_snn[s]], spikeRelu):
                si = spike_index[s]
                break
        #print(si)
    else:
        # send num_to_name_ann and layer_num+1 to partial model class
        SPLIT_LAYER = split_layer_num + 1
        si = spike_index[split_layer_num]

    if 'vgg' in arch:
        from models.vgg16_spiking import vggnet_partial
        model_partial = vggnet_partial(SPLIT_LAYER, num_to_name_ann, n_to_t_ann)
    elif 'alex' in arch:
        from models.alex_spiking import alexnet_partial
        model_partial = alexnet_partial(SPLIT_LAYER, num_to_name_ann, n_to_t_ann)
    elif 'svhn' in arch:
        from models.svhn_spike import svhn_partial
        model_partial = svhn_partial(SPLIT_LAYER, num_to_name_ann, n_to_t_ann)

    return si, model_partial


import ast
def plot_correlations(corr, out_dir, config, class_num=-1):
    num_layers,num_samples = corr.shape
    num_imgs = config.getint('num_imgs')
    if num_imgs < 0:
        num_imgs = num_samples

    layer_nums = ast.literal_eval(config['layer_nums'])
    layers = [int(i) for i in layer_nums]
    plot_layers = []
    for i in layers:
        if i <= num_layers:
            plot_layers.append(i)

    csv_dict = {}
    plt.figure()
    #Plot correlations
    for l in plot_layers:
        plt.plot(corr[l][0:num_imgs])
        csv_dict[str(l)] = corr[l][0:num_imgs]
    leg = [str(i) for i in plot_layers]

    plt.legend(leg)
    plt.grid()
    print('Plotting correlations..')
    plt.savefig(os.path.join(out_dir, 'correlation'+str(class_num)+'.png'), bbox_inches='tight')

    #### write the correlation values to a csv file
    pd.DataFrame(csv_dict).to_csv(os.path.join(out_dir, 'correlation'+str(class_num)+'.csv'))


from collections import Counter
def plot_histogram(container, max_acts, spike_config, out_dir, class_num=-1):
    print('[INFO] Plotting histogram of spiking activity..')
    time_window = spike_config.getint('time_window')
    acts = [container[key] for key in container]
    i = 0
    import csv
    with open(os.path.join(out_dir,'spike_hist'+str(class_num)+'.csv'), mode='w') as fl:
        writer = csv.writer(fl, delimiter=',')
        writer.writerow(['0 spikes', '1-10 spikes', '11-50 spikes', '> 50 spikes', 'total spikes'])
        arr = np.zeros((len(acts), 4))
        for a in acts:
            if i > 200:
                break
            a = (acts[i] / max_acts[i]) * time_window
            a = a.flatten()
            recounted = Counter(abs(a))
            sort_rcnt = {}
            for r in sorted(recounted):
                sort_rcnt[r] = recounted[r]
            x = [k for k in sort_rcnt.keys()]
            y = [v for v in sort_rcnt.values()]
            inactive = 0
            ten = 0
            twenty = 0
            rest = 0
            for k,v in sort_rcnt.items():
                if k == 0:
                    inactive += v
                if k > 0 and k <= 10:
                    ten += v
                elif k > 10 and k <= 50:
                    twenty += v
                elif k > 50:
                    rest += v
            arr[i, 0] = inactive
            arr[i, 1] = ten
            arr[i, 2] = twenty
            arr[i, 3] = rest
            writer.writerow([inactive, ten, twenty, rest, sum(y)])

            print('\nlayer {}'.format(i))
            print('# of inactive neurons: {}'.format(inactive))
            print('# neurons with 1-10 spikes: {}'.format(ten))
            print('# neurons with 11-50 spikes: {}'.format(twenty))
            print('# neurons with > 50 spikes: {}'.format(rest))
            i += 1




def plot_activity(container, max_acts, out_dir, class_num=-1):
    acts = [container[key] for key in container]

    plt.figure()
    i = 0
    mean, std = [], []
    for a in acts:
        a = a / max_acts[i]
        #print(a.shape)
        if i == 0:
            b = np.zeros(a.shape)
            b[np.nonzero(a)] = abs(a[np.nonzero(a)])
            std.append(np.std(b))
            mean.append(np.mean(b))
        else:
            std.append(np.std(a))
            mean.append(np.mean(a))

        i += 1
    layer_num = [i for i in range(len(container))]

    plt.plot(layer_num[1:], mean[1:], 'ro', label='mean')
    plt.plot(layer_num[1:], std[1:], 'c^', label='std')

    plt.title('Layer-wise average spiking activity percentage')
    plt.xlabel('layer number')
    plt.ylabel('ratio')
    plt.legend()
    plt.grid()
    print('Plotting activity..')
    plt.savefig(os.path.join(out_dir, 'activity'+str(class_num)+'.png'), bbox_inches='tight')

    #### write the mean spike activity factors to a csv file
    pd.DataFrame(np.asarray(mean)).to_csv(os.path.join(out_dir, 'activity'+str(class_num)+'.csv'))



def plot_mean_var(net, spike_net, out_dir):
    "wt, bias mean and var of the original model"

    wt_mean, wt_var = [], []
    bias_mean, bias_var = [], []
    layer_num, layer_num_b = [], []
    i = 1
    for m in net.modules():
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            with torch.no_grad():
                layer_num.append(i)
                wt_mean.append(m.weight.mean().cpu().numpy())
                wt_var.append(m.weight.var().cpu().numpy())
                if m.bias is not None:
                    layer_num_b.append(i)
                    bias_mean.append(m.bias.mean().cpu().numpy())
                    bias_var.append(m.bias.var().cpu().numpy())
                i += 1

    "wt, bias mean and var of the spiking model"
    wt_mean_s, wt_var_s = [], []
    bias_mean_s, bias_var_s = [], []
    layer_num_s, layer_num_b_s = [], []
    i = 1
    for m in spike_net.modules():
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            with torch.no_grad():
                layer_num_s.append(i)
                wt_mean_s.append(m.weight.mean().cpu().numpy())
                wt_var_s.append(m.weight.var().cpu().numpy())
                if m.bias is not None:
                    layer_num_b_s.append(i)
                    bias_mean_s.append(m.bias.mean().cpu().numpy())
                    bias_var_s.append(m.bias.var().cpu().numpy())
                i += 1

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(layer_num, wt_mean, 'ro', label='mean')
    plt.plot(layer_num, wt_var, 'c^', label='variance')
    plt.title('original model weights')
    plt.legend()

    plt.subplot(2, 2, 2)
    if len(bias_mean) > 0:
        plt.plot(layer_num_b, bias_mean, 'go', label='mean')
        plt.plot(layer_num_b, bias_var, 'b^', label='variance')
        plt.title('original model biases')
        plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(layer_num, wt_mean_s, 'ro', label='mean')
    plt.plot(layer_num, wt_var_s, 'c^', label='variance')
    plt.title('spike model weights')
    plt.legend()
    plt.xlabel('layer number')

    plt.subplot(2, 2, 4)
    if len(bias_mean_s) > 0:
        plt.plot(layer_num_b_s, bias_mean_s, 'go', label='mean')
        plt.plot(layer_num_b_s, bias_var_s, 'b^', label='variance')
        plt.title('spike model biases')
        plt.legend()
        plt.xlabel('layer number')
    plt.grid()

    plt.savefig(os.path.join(out_dir, 'mean_var.png'), bbox_inches='tight')



# rand_val corresponds to vmem
# in_val/max_in_val corresponds to threshold
def condition(rand_val, in_val, abs_max_val, MFR=1):
    if rand_val <= (abs(in_val) / abs_max_val) * MFR:
        return (np.sign(in_val))
    else:
        return 0

def poisson_spikes(pixel_vals, MFR):
    " MFR = maximum firing rate "
    " Use when GPU is short of memory. Slower implementation"
    out_spikes = np.zeros(pixel_vals.shape)
    for b in range(pixel_vals.shape[0]):
        random_inputs = np.random.rand(pixel_vals.shape[1],pixel_vals.shape[2], pixel_vals.shape[3])
        single_img = pixel_vals[b,:,:,:]
        max_val = np.amax(abs(single_img)) # note: shouldn't this be max(abs(single_img)) ??
        vfunc = np.vectorize(condition)
        out_spikes[b,:,:,:] = vfunc(random_inputs, single_img, max_val, MFR)
    return out_spikes


def poisson_spikes_torch(images, MFR=1, device='cuda:0'):
    ''' creates poisson spikes from input images '''
    ''' shape of input and output: BCHWT. Faster. '''
    #images = images.cpu()
    random_vals = torch.rand(images.size(), device=device)
    images_max = images.cpu().numpy().max(axis=(1,2,3), keepdims=True)
    images_max = torch.from_numpy(images_max).to(device)
    ratio = (abs(images) / images_max) * MFR
    #print(ratio.size())
    random_vals = torch.where(random_vals <= ratio, torch.tensor(1, device=device), \
            torch.tensor(0, device=device))
    return random_vals


def sanity_check(net, spike_net, max_acts):

    num = 0
    num_to_type = {}
    for name, module in net.named_modules():
        if type(module) == nn.Conv2d or type(module) == nn.Linear:
            num_to_type[num] = module
            num += 1

    num = 0
    i = 0
    for name, module in spike_net.named_modules():
        if type(module) == nn.Conv2d or type(module) == nn.Linear:
            if not(torch.all(module.weight.data == num_to_type[num].weight.data)):
                print ('weights dont match at layer {}'.format(num))

            if module.bias is not None:
                if not(torch.all(module.bias.data == num_to_type[num].bias.data / max_acts[i])):
                    print ('biases dont match at layer {}'.format(num))

            num += 1
            i += 1


def createSpikingModel(net, arch, num_classes, spike_config, thresholds, max_acts, \
        device='cuda:0', out_dir=None):

    ''' check if model has BN layers '''
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            print('model {} has BN layers. Can\'t spikify. Exiting...'.format(arch))
            exit()

    clamp_slope = spike_config.getint('clamp_slope')
    reset = spike_config['reset']
    unity_vth = spike_config.getboolean('unity_vth')

    num = 0
    num_to_type = {}
    for name, module in net.named_modules():
        #print(name)
        if type(module) == nn.Conv2d or type(module) == nn.Linear or \
            type(module) == nn.AvgPool2d or type(module) == nn.AdaptiveAvgPool2d:
            num_to_type[num] = module

            if unity_vth and (type(module) == nn.Conv2d or type(module) == nn.Linear):
                thresholds[num] = 1

            num += 1

    spike_net = None

    if 'vgg_cifar10' in arch:
        from models import vgg16_spiking
        spike_net = vgg16_spiking.vgg16_spike(thresholds, device, clamp_slope, num_classes, reset)

    elif 'mobnet_cif10_mod' in arch:
        from models import mobilenet_mod_spiking
        spike_net = mobilenet_mod_spiking.mobilenet_mod_spike(thresholds, device, clamp_slope, num_classes, reset)

    elif 'mobnet_cif100' in arch:
        from models.mobnet_cif100_nobn_spike import mobilenet_cif100_nobn_spike
        spike_net = mobilenet_cif100_nobn_spike(thresholds, device, clamp_slope, reset)

    elif 'vgg_cif100' in arch:
        from models.vgg_cif100_spike import vgg13_nobn_spike
        spike_net = vgg13_nobn_spike(thresholds, device, clamp_slope, reset)

    elif 'svhn' in arch:
        from models.svhn_spike import svhn_spike
        spike_net = svhn_spike(thresholds, device, clamp_slope, reset)

    elif 'alex' in arch:
        from models.alex_spiking import alexnet_spiking
        spike_net = alexnet_spiking(thresholds, device, clamp_slope, reset)

    elif 'lenet5' in arch:
        from models.lenet5 import lenet5_spiking
        spike_net = lenet5_spiking(thresholds, device, clamp_slope, reset)


    ####### copy and adjust weights #######
    if unity_vth:
    # when all vth is normalized to 1 wts normalized by max_acts
        j = 0
        layer_num = 0
        for nm, spk_layer in spike_net.named_modules():
            if isinstance(spk_layer, torch.nn.Conv2d) or isinstance(spk_layer, nn.Linear) or \
                isinstance(spk_layer, nn.AvgPool2d) or isinstance(spk_layer, nn.AdaptiveAvgPool2d):

                if isinstance(spk_layer, torch.nn.Conv2d) or isinstance(spk_layer, nn.Linear):
                    L = num_to_type[layer_num]

                    scale = max_acts[j] / max_acts[j+1]
                    spk_layer.weight = torch.nn.Parameter(L.weight * scale)

                    if spk_layer.bias is not None:
                        temp_b = L.bias / max_acts[j+1]
                        spk_layer.bias = torch.nn.Parameter(temp_b)

                    #print('{}, max act indices: {},{}'.format(nm, j, j+1))

                layer_num += 1
                j += 1

    else:
        ## the following works for thresholds = max_acts and same weights
        ## as original ANN
        num = 0
        for name, module in spike_net.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, nn.Linear) or \
                isinstance(module, nn.AvgPool2d) or isinstance(module, nn.AdaptiveAvgPool2d):

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    module.weight.data = num_to_type[num].weight.data.clone()
                    if num_to_type[num].bias is not None and module.bias is not None:
                        temp = num_to_type[num].bias.data.clone()
                        module.bias.data = torch.nn.Parameter(temp / max_acts[num])
                num += 1

    ####### copy and adjust weights (ends) #######

    return spike_net
