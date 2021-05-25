from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import datetime

from tqdm import tqdm
from models import *
import utils
import pdb
from attack_methods import Attack_None, Attack_PGD, Attack_GAT
import numpy as np

torch.manual_seed(101)
torch.cuda.manual_seed(101)
np.random.seed(101)

parser = argparse.ArgumentParser(
    description='Dual-label Geometry Dispersion Adversarial Training')

parser.register('type', 'bool', utils.str2bool)

parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--attack', default=True, type='bool', help='attack')
parser.add_argument('--model_dir', default='./checkpoint/',type=str, help='model path')
parser.add_argument('--init_model_pass',
                    default='latest',
                    type=str,
                    help='init model pass')

parser.add_argument('--attack_method',
                    default='pgd',
                    type=str,
                    help='adv_mode (natural, fgsm or pdg)')
parser.add_argument('--attack_method_list', type=str)

parser.add_argument('--log_step', default=20, type=int, help='log_step')

# dataset dependent
parser.add_argument('--epoch', default=100, type=int, help='num epoch')
parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')  # concat cascade
parser.add_argument('--batch_size_test',
                    default=100,
                    type=int,
                    help='batch size for testing')
parser.add_argument('--image_size', default=32, type=int, help='image size')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--save_name', default='wide-res-dual-geometry', type=str, help='save model name')

args = parser.parse_args()

if args.dataset == 'cifar10':
    print('------------cifar10---------')
    args.num_classes = 10
    args.image_size = 32
elif args.dataset == 'cifar100':
    print('----------cifar100---------')
    args.num_classes = 100
    args.image_size = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

# Data
print('==> Preparing data..')

if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])

if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root='./cifar10',
                                           train=False,
                                           download=True,
                                           transform=transform_test)
elif args.dataset == 'cifar100':
    testset = torchvision.datasets.CIFAR100(root='./cifar100',
                                            train=False,
                                            download=True,
                                            transform=transform_test)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.batch_size_test,
                                         shuffle=False,
                                         num_workers=2)

print('==> Building model..')

def getNetwork(args):
    num_classes = args.num_classes
    if (args.net_type == 'wide-resnet'):
        net = WideResNet(depth=args.depth, widen_factor=args.widen_factor,num_classes=num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    basic_net, net_name = getNetwork(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True
    basic_net = basic_net.to(device)

# configs
config_natural = {'train': False}

config_fgsm = {
    'train': False,
    'targeted': False,
    'epsilon': 8.0 / 255 * 2
    'num_steps': 1,
    'step_size': 8.0 / 255 * 2,
    'random_start': True
}

config_pgd = {
    'train': False,
    'targeted': False,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 20,
    'step_size': 2.0 / 255 * 2,
    'random_start': True,
    'loss_func': torch.nn.CrossEntropyLoss(reduction='none')
}

def test(epoch, net):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    loss_list  = []

    iterator = tqdm(testloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        pert_inputs = inputs.detach()

        outputs = net(pert_inputs, targets, batch_idx=batch_idx)

        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        loss_list.append(loss.item())

        duration = time.time() - start_time

        _, predicted = outputs.max(1)
        batch_size = targets.size(0)
        total += batch_size
        correct_num = predicted.eq(targets).sum().item()
        correct += correct_num
        acc_loc = float(correct_num) / batch_size
        iterator.set_description(str(f'{100*acc_loc:.2f}'))

        if batch_idx % args.log_step == 0:
            print(
                ">| step %d, duration %.2f, test  acc %.2f, avg-acc %.2f, loss %.2f"
                % (batch_idx, duration, 100. * correct_num / batch_size,
                   100. * correct / total, test_loss / total))

    acc = 100. * correct / float(total)
    print('Val acc:', acc)

    return acc

attack_list = args.attack_method_list.split('-')
attack_num = len(attack_list)

for attack_idx in range(attack_num):

    args.attack_method = attack_list[attack_idx]
    if args.attack_method == 'natural':
        print('-----natural non-adv mode -----')
        net = Attack_None(basic_net, config_natural)
    elif args.attack_method.upper() == 'FGSM':
        print('-----FGSM adv mode -----')
        net = Attack_PGD(basic_net, config_fgsm)
    elif args.attack_method.upper() == 'PGD':
        print('-----PGD adv mode -----')
        net = Attack_PGD(basic_net, config_pgd)
    else:
        raise Exception(
            'Should be a valid attack method. The specified attack method is: {}'
            .format(args.attack_method))

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume and args.init_model_pass != '-1':
        # Load checkpoint.
        print('==> Resuming from checkpoint..')

        save_point = args.model_dir+args.dataset+os.sep        
        f_path = save_point + args.save_name+f'-{args.epoch}.t7'
        f_path_latest = save_point + args.save_name+f'-latest.t7'
        
        if not os.path.isdir(args.model_dir):
            print('train from scratch: no checkpoint directory or file found')
        elif args.init_model_pass == 'latest' and os.path.isfile(f_path_latest):
            checkpoint = torch.load(f_path_latest)      
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            print('resuming from epoch %s in latest' % start_epoch)
        elif os.path.isfile(f_path):
            checkpoint = torch.load(f_path)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            print('resuming from epoch %s' % start_epoch)
        elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
            print('train from scratch: no checkpoint directory or file found')

    criterion = nn.CrossEntropyLoss()

    test(0, net)
print(args.save_name+' Finished!')
