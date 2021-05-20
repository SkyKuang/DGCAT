'''Train Adversarially Robust Models with Dual-label Geometry Dispersion'''
from __future__ import print_function
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from torch.autograd.gradcheck import zero_gradients
import copy
from torch.autograd import Variable

import os
import argparse
import datetime
import pdb
from tqdm import tqdm
from models import *

import utils
from utils import get_hms
from attack_methods import Attack_GAT


torch.manual_seed(233)
torch.cuda.manual_seed(233)
np.random.seed(233)

torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='Dual-label Geometry Dispersion Adversarial Training')

# add type keyword to registries
parser.register('type', 'bool', utils.str2bool)

parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--adv_mode',
                    default='dual-label-geometry-dispersion',
                    type=str,
                    help='adv_mode (dual-label-geometry-dispersion)')
parser.add_argument('--model_dir', default='./checkpoint/',type=str, help='model path')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass (-1: from scratch; K: checkpoint-K)')
parser.add_argument('--max_epoch',
                    default=200,
                    type=int,
                    help='max number of epochs')
parser.add_argument('--save_epochs', default=150, type=int, help='save period')
parser.add_argument('--decay_epoch1',
                    default=80,
                    type=int,
                    help='learning rate decay epoch one')
parser.add_argument('--decay_epoch2',
                    default=120,
                    type=int,
                    help='learning rate decay point two')
parser.add_argument('--decay_rate',
                    default=0.1,
                    type=float,
                    help='learning rate decay rate')
parser.add_argument('--batch_size_train',
                    default=100,
                    type=int,
                    help='batch size for training')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    help='momentum (1-tf.momentum)')
parser.add_argument('--weight_decay',
                    default=2e-4,
                    type=float,
                    help='weight decay')
parser.add_argument('--log_step', default=50, type=int, help='log_step')

# number of classes and image size will be updated below based on the dataset
parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--image_size', default=32, type=int, help='image size')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')  # concat cascade
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--save_name', default='wide-res-dual-geometry', type=str, help='save model name')

args = parser.parse_args()

if args.dataset == 'cifar10':
    print('------------cifar10---------')
    # args.num_classes = 10
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
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./cifar10',
                                            train=True,
                                            download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./cifar10',
                                           train=False,
                                           download=True,
                                           transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./cifar100',
                                             train=True,
                                             download=True,
                                             transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./cifar100',
                                            train=False,
                                            download=True,
                                            transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size_train,
                                          shuffle=True,
                                          num_workers=2)

testloader = torch.utils.data.DataLoader(testset,
                                          batch_size=64,
                                          shuffle=True,
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

def print_para(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data)
        break

basic_net = basic_net.to(device)

# config for feature scatter
config_feature_scatter = {
    'train': True,
    'epsilon': 8.0 / 255 * 1,
    'num_steps': 1,
    'step_size': 8.0 / 255 * 1,
    'random_start': True,
    'ls_factor': 0.5,
    'alpha':0.4
}

if args.adv_mode.lower() == 'dual-label-geometry-dispersion':
    print('-----Dual-Label-Geometry-Dispersion -----')
    net = Attack_GAT(basic_net, config_feature_scatter)
else:
    print('-----OTHER_ALGO mode -----')
    raise NotImplementedError("Please implement this algorithm first!")

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)
# print(net.parameters())
if args.resume and args.init_model_pass != '-1':
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    save_point = args.model_dir+args.dataset+os.sep
    f_path = save_point + args.save_name+f'-latest.t7'
    print(f_path)
    if not os.path.isdir(args.model_dir):
        print('train from scratch: no checkpoint directory or file found')
    elif args.init_model_pass == 'latest' and os.path.isfile(f_path):
        checkpoint = torch.load(f_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        print('resuming from epoch %s in latest' % start_epoch)
    elif os.path.isfile(f_path):
        checkpoint = torch.load(f_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        print('resuming from epoch %s' % (start_epoch - 1))
    elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
        print('train from scratch: no checkpoint directory or file found')

start_time = time.time()
def train_fun(epoch, net):
    print('\n Epoch: %d' % epoch)
    net.train()

    correct = 0
    total = 0

    # update learning rate
    if epoch < args.decay_epoch1:
        lr = args.lr
    elif epoch < args.decay_epoch2:
        lr = args.lr * args.decay_rate
    else:
        lr = args.lr * args.decay_rate * args.decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    def get_acc(outputs, targets):
        _, predicted = outputs.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        acc = 1.0 * correct / total
        return acc

    iterator = tqdm(trainloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
     
        adv_acc = 0
        optimizer.zero_grad()
        # forward
        outputs_adv, outputs_nat, loss, nat_loss, adv_loss  = net(inputs.detach(), targets)
        
        optimizer.zero_grad()
        total_loss = loss
        total_loss = total_loss.mean()
        total_loss.backward()
        
        optimizer.step()

        if batch_idx % args.log_step == 0:
            adv_acc = get_acc(outputs_adv, targets)
            nat_acc = get_acc(outputs_nat, targets)
        
            duration = time.time() - start_time
            h,m,s=get_hms(duration)
            print('\r')
            inform = "| Step %3d, lr %.4f, time %d:%02d:%02d, loss %.4f, nat acc %.2f,adv acc %.2f" % (batch_idx, lr,h,m,s, loss, 100 * nat_acc, 100 * adv_acc)
            iterator.set_description(str(inform))

    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    save_point = args.model_dir+args.dataset+os.sep
    if not os.path.isdir(save_point):
        os.mkdir(save_point)
    if epoch % args.save_epochs == 0:
        correct = 0
        total = 0
        net.eval()
        for batch_idx, (inputs, targets) in enumerate(testloader):
     
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_nat, _ = net(inputs, targets, attack=False)

            nat_acc = get_acc(outputs_nat, targets)
            correct += nat_acc
            total += 1
        
        print(f'| Test acc:{100.0*correct/total:.4f}')
        print('| Saving...')
        state = {
            'net': net.state_dict(),
        }      
        f_path = save_point + args.save_name+f'-{epoch}.t7'
        print(f_path)
        torch.save(state, f_path)

    if epoch >= 0:
        print(f'| Saving {args.save_name} latest @ {epoch} %s...\r' )
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        f_path = save_point + args.save_name+f'-latest.t7'
        torch.save(state, f_path)

for epoch in range(start_epoch, args.max_epoch):
    train_fun(epoch, net)
print(args.save_name+' Finished !')

