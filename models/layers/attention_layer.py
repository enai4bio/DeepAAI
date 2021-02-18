#!/usr/bin/env
# coding:utf-8
"""
Created on 2020/7/30 10:05

base Info
"""
__author__ = 'xx'
__version__ = '1.0'



import torch
import torch.nn as nn
import torch.nn.functional as F


class Att2One(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(Att2One, self).__init__()

        self.linear_trans = nn.Linear(input_dim, hidden_dim)

        self.linear_q = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, ft_mat):
        '''
        :param ft_mat:  batch, n_channel, ft
        :return:
        '''
        w_mat = torch.tanh(self.linear_trans(ft_mat))
        w_mat = self.linear_q(w_mat)
        # print(w_mat[0])
        w_mat = F.softmax(w_mat, dim=1)  # batch n_channel 1
        # print(w_mat.shape, w_mat[0])
        ft_mat = torch.sum(ft_mat * w_mat, dim=1) # batch 1 ft
        ft_mat = ft_mat.squeeze()
        # print(ft_mat.size())

        return ft_mat


class Att2One2Type(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(Att2One2Type, self).__init__()

        self.linear_trans_type_a = nn.Linear(input_dim, hidden_dim)

        self.linear_trans_type_b = nn.Linear(input_dim, hidden_dim)

        self.linear_q = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, ft_a, ft_b):
        '''
        :param ft_a:  batch, n_channel_a, ft
        :param ft_b:  batch, n_channel_b, ft
        :return:
        '''
        w_mat_a = torch.tanh(self.linear_trans_type_a(ft_a))
        w_mat_b = torch.tanh(self.linear_trans_type_b(ft_b))
        w_mat = torch.cat([w_mat_a, w_mat_b], dim=1)

        w_mat = self.linear_q(w_mat)
        # print(w_mat[0])
        w_mat = F.softmax(w_mat, dim=1)  # batch n_channel 1
        # print(w_mat.shape, w_mat[0])

        raw_fr_mat = torch.cat([ft_a, ft_b], dim=1)
        ft_mat = torch.sum(raw_fr_mat * w_mat, dim=1) # batch 1 ft
        ft_mat = ft_mat.squeeze()
        # print(ft_mat.size())

        return ft_mat


if __name__ == '__main__':
    # ft_mat = torch.rand([10, 5, 12])
    # att = Att2One(12)
    # att(ft_mat)


    ft_mat_a = torch.rand([10, 5, 12])
    ft_mat_b = torch.rand([10, 8, 12])

    att = Att2One2Type(12)
    att(ft_mat_a, ft_mat_b)
