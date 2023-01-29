from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CSA(nn.Module):
  def __init__(self, num_bands: int=156):
    super(CSA, self).__init__()
    self.num_bands = num_bands

  def forward(self, input, target):
    normalize_r = (input/torch.sum(input, dim=0)) 
    normalize_t = (target/torch.sum(target, dim=0))
    mult = torch.multiply(normalize_r,normalize_t).sum()
    angle= 1-mult.sum()
    return angle
