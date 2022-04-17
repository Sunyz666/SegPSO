import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.HA_units import HA_unit
from torch.autograd import Variable

class HA_module(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, delta=0.5, hop=3, stride=1):
        super(HA_module, self).__init__()
        channels = 512
        print('hop: {}  delta: {}'.format(hop, delta))
        self.HA = nn.Sequential(
                    nn.Conv2d(in_channels=2048, out_channels=channels,
                        kernel_size=3, stride=stride, padding=1),
                    BatchNorm(channels),
                    HA_unit(in_channels=channels, inter_channels=channels, out_channels=channels, 
                            delta=delta, hop=hop)
                )
        
        self.top_conv = nn.Sequential(
                     nn.Conv2d(in_channels=channels, out_channels=256,
                        kernel_size=1, stride=1, padding=0),
                     BatchNorm(256),
                     nn.Dropout(0.5),
                     nn.ReLU()
        )
        
        self._init_weight()
        
    def forward(self, x):
        x = self.HA(x)
        x = self.top_conv(x)
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()