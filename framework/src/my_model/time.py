import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import pdb
import sys
import os 
import math
from einops import rearrange, reduce, repeat

from .blocks import make_mlp, SkipCMLP
from .freq  import poc_encoding


class TimeCodes(nn.Module):

    def __init__(self, frame_num, t_dim):
        super().__init__() 

        # time codes
        t_codes = torch.zeros(
                    (frame_num, t_dim), dtype=torch.float32)

        nn.init.kaiming_uniform_(t_codes, mode='fan_in', 
                                 a=math.sqrt(5)) 
        self.register_parameter('t_codes', 
                        torch.nn.Parameter(t_codes))

    def forward(self, t_idx):

        assert self.t_codes is not None
        assert t_idx.ndim==1

        # n c
        tz = self.t_codes[t_idx] 

        return tz  






