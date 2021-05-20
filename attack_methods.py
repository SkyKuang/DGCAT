import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import utils
import math
import pdb
import pickle
from loss import Geometry_loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attack_None(nn.Module):
    def __init__(self, basic_net, config):
        super(Attack_None, self).__init__()
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.basic_net = basic_net
        print(config)

    def forward(self, inputs, targets=None, attack=None, batch_idx=-1):
        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()
        outputs, _ = self.basic_net(inputs)
        return outputs

class Attack_PGD(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_PGD, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction='none') if 'loss_func' not in config.keys(
            ) else config['loss_func']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        # self.box_type = 'black'

        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0,
                black_net=None):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert black_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.attack_net))
            aux_net = black_net
            print('--------black attack-----------')

        aux_net.eval()

        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            loss = self.loss_func(logits, y_tensor_adv)
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]

        return logits_pert

class Attack_GAT(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_GAT, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']

        self.criterion_G = Geometry_loss()
        self.alpha = config['alpha']
        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs, _ = self.basic_net(inputs)
            return outputs, None
        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
 

        aux_net.eval()
        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size

        step_sign = 1.0

        x = inputs.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pred_ori, fea_ori = aux_net(inputs)

        iter_num = self.num_steps

        for i in range(iter_num):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred, fea = aux_net(x)
            
            adv_loss = 0
            # ~~~~~~~~~ geometry ~~~~~~~~~~~~~~
            adv_loss += self.criterion_G(fea, fea_ori) 
                        
            aux_net.zero_grad()
            adv_loss.backward(retain_graph=True)
            x_adv = x.data + self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        logits_pred, fea = self.basic_net(x)

        self.basic_net.zero_grad()

        loss_ce = nn.CrossEntropyLoss()
        true_loss = loss_ce(logits_pred, targets)

        adv_targets = targets[torch.randperm(targets.size(0))]
        adv_loss = loss_ce(logits_pred, adv_targets)

        loss = (1- self.alpha)*true_loss + self.alpha*adv_loss

        return logits_pred, logits_pred_ori, loss, 0, 0
