#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/23 15:33

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from models.layers.model_utils import get_laplace_mat


class GCNConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 dropout=0.6,
                 bias=True
                 ):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.bias = bias
        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels)
        )
        nn.init.xavier_normal_(self.weight)
        if bias is True:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, node_ft, adj_mat):
        laplace_mat = get_laplace_mat(adj_mat, type='sym')
        node_state = torch.mm(laplace_mat, node_ft)
        node_state = torch.mm(node_state, self.weight)
        if self.bias is not None:
            node_state = node_state + self.bias

        return node_state


