'''Some utility functions
'''
import os
import sys
import time
import datetime
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
import random
import scipy.io

import torch
from torch.optim import Optimizer

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

