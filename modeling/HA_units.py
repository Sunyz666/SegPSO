import torch
import os
import sys
import pdb
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
    
class sparse_hop(nn.Module):
    def __init__(self, inter_channels, delta=0.5, hop=3):
        super(sparse_hop, self).__init__()
        self.inter_channels = inter_channels
        self.delta = torch.nn.Parameter(torch.FloatTensor([delta])) #Auto assign device
        self.hop = hop
        
        self.hop_conv = nn.ModuleList([])
        for i in range(self.hop):
            self.hop_conv.append(nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels,
                kernel_size=1, stride=1, padding=0))  

    def forward(self, initial_w, nodes):
        batch_size, w, h = nodes.size(0), nodes.size(2), nodes.size(3)
        nodes = nodes.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        
        b0 = torch.sigmoid(initial_w)
        b0 = (b0 - self.delta).ge(0.).float()
        bh = b0

        hops = []
        for i in range(self.hop):
            t = torch.matmul(F.softmax(bh.mul(initial_w), dim=-1), nodes)
            t = t.permute(0, 2, 1).contiguous()
            t = t.view(initial_w.size(0), self.inter_channels, w, h)
            t = self.hop_conv[i](t)
            hops.append(t)
            if i != self.hop - 1:
                bh = bh.matmul(b0).float()
        return hops
    
class HA_unit(nn.Module):
    def __init__(self,  in_channels, inter_channels=256, out_channels=None,
                 delta=0.5, hop=3, dropout = 0.05, BatchNorm=nn.BatchNorm2d):
        super(HA_unit, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.hop = hop
        self.delta = delta
        
        self.hop_block = sparse_hop(self.inter_channels, delta=self.delta, hop=self.hop)
        
        if out_channels == None:
            self.out_channels = in_channels
            
        self.w1_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                kernel_size=1, stride=1, padding=0),
            BatchNorm(self.inter_channels)
        )
        self.w2_conv = self.w1_conv
        
        self.node_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
            kernel_size=1, stride=1, padding=0)
        
        self.fuse_conv =  nn.Sequential(
            nn.Conv2d((self.hop)*self.inter_channels, self.inter_channels, kernel_size=1, groups=1, padding=0),
            BatchNorm(self.inter_channels)
        )
        
        self.res_conv = nn.Sequential(
            nn.Conv2d((2)*self.inter_channels, self.out_channels, kernel_size=1, padding=0),
            BatchNorm(self.out_channels),
            nn.Dropout2d(dropout),
            )
    
    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        
        w1 = self.w1_conv(x).view(batch_size, self.inter_channels, -1)#，n, c, h*w
        w1 = w1.permute(0, 2, 1)
        w2 = self.w2_conv(x).view(batch_size, self.inter_channels, -1)
        
        nodes = self.node_conv(x)
        
        initial_w = torch.matmul(w1, w2)
        initial_w =  (self.inter_channels**-.5) * initial_w
        hops = self.hop_block(initial_w, nodes)
        
        x_p = torch.cat(hops, dim=1)
        x_p = self.fuse_conv(x_p)
        x_p = self.res_conv(torch.cat((x, x_p), dim=1))
        return x_p
    
    
    
    
    

    
