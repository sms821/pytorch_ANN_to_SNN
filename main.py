import numpy as np
import os
import configparser
import argparse
import pprint

import torch
#from torchsummary import summary
#
#from models.mobnet_cif100 import mobilenet_cif100
#from models.mobnet_cif100_nobn import mobilenet_cif100_nobn
#from models.vgg_cif100 import vgg13_bn, vgg13_nobn
#from models.svhn import svhn
#from models.alex import alexnet
#from models.alex import AlexNet
#from models.len5_mlp import lenet5 #, mnist_mlp
from utils import *
#from spiking import *

def main():
    parser = argparse.ArgumentParser(description='Deep Learning SNN simulation')
    parser.add_argument('--config-file', default='config.ini')
    args = parser.parse_args()
    print (args)

    config = configparser.ConfigParser()
    config.read(args.config_file)
    pprint.pprint({section: dict(config[section]) for section in config.sections()})
    print

    defaults = config['DEFAULT']
    device = defaults['device']
    app_name = defaults['app_name']

    print ('[INFO] Simulating spiking {}'.format(app_name))
    if not os.path.isdir(app_name):
        os.mkdir(app_name)
    out_dir = app_name

    org_model = config['original model']
    num_classes = org_model.getint('num_classes')
    model_path = org_model['model_path']
    file_name = org_model['file_name']
    batch_size = org_model.getint('batch_size')
    class_num = org_model.getint('class')

    " Load the original model "
    net = None
    if 'vgg_cifar10' in org_model['arch']:
        from models import VGG_16_cifar10
        net = VGG_16_cifar10
    if 'mobnet_cif10_mod' in org_model['arch']:
        from models import MobileNet_mod
        net = MobileNet_mod()
    if 'mobnet_cif100' in org_model['arch']:
        net = mobilenet_cif100()
    if 'vgg_cif100' in org_model['arch']:
        net = vgg13_bn()
    if 'svhn' in org_model['arch']:
        net = svhn()
    if 'alex' in org_model['arch']:
        net = alexnet()
    if 'lenet5' in org_model['arch']:
        net = lenet5()

    print(net)
    state = None
    state, net = load_model(net, model_path, file_name)

    net = net.to(device)

    " Load the dataset "
    testloader, img_size = None, None
    data_config = config['dataset']
    data_dir = data_config['data_dir']
    if data_config['dataset'] == 'cifar10':
        trainloader, testloader = load_cifar10(data_dir, org_model['arch'], batch_size, class_num)
        img_size = (-1,3,32,32)

    if data_config['dataset'] == 'imagenet':
        trainloader, testloader = load_imagenet(data_dir, batch_size, shuffle=False)
        img_size = (-1,3,224,224)

    if data_config['dataset'] == 'cifar100':
        trainloader, testloader = load_cifar100(data_dir, org_model['arch'], batch_size, class_num)
        img_size = (-1,3,32,32)

    if data_config['dataset'] == 'svhn':
        trainloader, testloader = load_svhn(data_dir, org_model['arch'], batch_size, class_num)
        img_size = (-1,3,32,32)

    if data_config['dataset'] == 'mnist':
        trainloader, testloader = load_mnist(data_dir, org_model['arch'], batch_size, class_num)
        img_size = (-1,1,28,28)

    " Tasks to do "
    tasks = config['functions']
    " validate original model "
    if tasks.getboolean('validate'):
        validate(net, testloader, device)

    " fold back BN layers if any "
    new_net = None
    remove_bn = tasks.getboolean('remove_bn')
    if remove_bn:
        if has_bn(net):
            if 'mobnet_cif10_mod' in org_model['arch']:
                from models import MobileNet_mod_nobn
                new_net = MobileNet_mod_nobn()
            elif 'mobnet_cif100' in org_model['arch']:
                new_net = mobilenet_cif100_nobn()
            elif 'vgg_cif100' in org_model['arch']:
                new_net = vgg13_nobn()

            new_net = merge_bn(net, new_net)
            new_net = new_net.to(device)
            print(new_net)
            print('Validating model after folding back BN layers...')
            validate(new_net, testloader, device)
            save_model(new_net, state, out_dir, 'nobn_'+file_name)
        else:
            print('model has no BN layers')

    " use model with folded BN "
    use_nobn = tasks.getboolean('use_nobn')
    if use_nobn:
        if 'mobnet_cif10_mod' in org_model['arch']:
            from models import MobileNet_mod_nobn
            net = MobileNet_mod_nobn()
        elif 'mobnet_cif100' in org_model['arch']:
            net = mobilenet_cif100_nobn()
        elif 'vgg_cif100' in org_model['arch']:
            net = vgg13_nobn()
        state, net = load_model(net, out_dir, 'nobn_'+file_name)
        net = net.to(device)

    " validate model with folded BN "
    validate_nobn = tasks.getboolean('validate_nobn')
    if not remove_bn and validate_nobn:
        if not has_bn(net):
            print('Validating no_bn model...')
            validate(net, testloader)
        else:
            print('model has BN layers!! Exiting..')
            exit()

    " compute thresholds "
    spike_config = config['spiking']
    percentile = spike_config.getfloat('percentile')
    if spike_config.getboolean('compute_thresholds'):
        compute_thresholds(net, testloader, out_dir, percentile, device)

    " convert ann to snn "
    thresholds, max_acts = None, None
    if spike_config.getboolean('convert_to_spike'):
        from spiking import createSpikingModel

        thresholds = np.loadtxt(os.path.join(out_dir, 'thresholds.txt'))
        max_acts = np.loadtxt(os.path.join(out_dir, 'max_acts.txt'))

        spike_net = createSpikingModel(net, org_model['arch'], num_classes, spike_config, \
                torch.from_numpy(thresholds), max_acts, device, out_dir)
        #print(spike_net.state_dict().keys())

        print(spike_net)

    " simulate snn "
    if spike_config.getboolean('simulate_spiking'):
        from spiking import simulate_spike_model

        thresholds = np.loadtxt(os.path.join(out_dir, 'thresholds.txt'))
        max_acts = np.loadtxt(os.path.join(out_dir, 'max_acts.txt'))

        thresholds = torch.from_numpy(thresholds).to(device)
        max_acts = torch.from_numpy(max_acts).to(device)

        sbi, model_partial = None, None
        # sbi: spike buffer index
        if spike_config.getboolean('hybrid_model'):
            from spiking import createSpikingModel, create_partial_model

            spike_net = createSpikingModel(net, org_model['arch'], num_classes, spike_config, \
                    thresholds, max_acts, device, out_dir)
            split_layer = spike_config.getint('split_layer')
            sbi, model_partial = create_partial_model(split_layer, net, spike_net, org_model['arch'])
            print(model_partial)

        simulate_spike_model(net, org_model['arch'], testloader, config, thresholds.float(), \
                max_acts, num_classes, img_size, sbi, model_partial, device)

    if class_num < 0:
        class_num = ''
    " plot correlations "
    if spike_config.getboolean('plot_correlations'):
        from spiking import plot_correlations

        corr = np.load(os.path.join(out_dir, 'layerwise_corr'+str(class_num)+'.npy'))
        plot_config = config['plotting']
        plot_correlations(corr, out_dir, plot_config, class_num)

    " plot activity "
    if spike_config.getboolean('plot_activity'):
        from spiking import plot_activity

        container = np.load(os.path.join(out_dir, 'layerwise_acts'+str(class_num)+'.npz'))
        max_acts = np.loadtxt(os.path.join(out_dir, 'max_acts.txt'))
        plot_activity(container, max_acts, out_dir, class_num)

    " plot histogram of spikes "
    if spike_config.getboolean('plot_histogram'):
        from spiking import plot_histogram

        container = np.load(os.path.join(out_dir, 'layerwise_acts'+str(class_num)+'.npz'))
        max_acts = np.loadtxt(os.path.join(out_dir, 'max_acts.txt'))
        plot_histogram(container, max_acts, spike_config, out_dir, class_num)


if __name__ == '__main__':
    main()
