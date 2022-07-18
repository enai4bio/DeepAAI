#!/usr/bin/env
# coding:utf-8

import torch.nn.functional as F
import torch.nn as nn
import torch


class BasicBlock2D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, res_connect=True):
        super(BasicBlock2D, self).__init__()
        self.res_connect = res_connect
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel//2, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        if self.res_connect is True:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        residual = self.residual_function(x)
        if self.res_connect:
            residual += self.shortcut(x)
        residual = F.relu(residual)
        return residual


class ResPPIReg(nn.Module):
    def __init__(self,
                 amino_ft_dim,
                 max_antibody_len,
                 max_virus_len,
                 h_dim=512,
                 dropout=0.3,
                 ):
        super(ResPPIReg, self).__init__()
        self.h_dim = h_dim
        self.amino_ft_dim = amino_ft_dim
        self.max_antibody_len = max_antibody_len
        self.max_virus_len = max_virus_len
        self.dropout = dropout
        self.mid_channels = 16
        self.out_linear1 = nn.Linear(self.mid_channels * self.amino_ft_dim * 2, self.h_dim)
        self.out_linear2 = nn.Linear(self.h_dim, 1)

        self.res_net = nn.Sequential(
            BasicBlock2D(in_channel=1, out_channel=self.mid_channels, res_connect=True),
            BasicBlock2D(in_channel=self.mid_channels, out_channel=self.mid_channels, res_connect=False),
            BasicBlock2D(in_channel=self.mid_channels, out_channel=self.mid_channels, res_connect=True),
            # BasicBlock2D(in_channel=64, out_channel=64, res_connect=False),
            # BasicBlock2D(in_channel=64, out_channel=64, res_connect=True),
        )
        self.activation = nn.ELU()

    def forward(self, batch_antibody_onehot_ft, batch_virus_onehot_ft):
        '''
        :param batch_antibody_ft:   tensor    batch, antibody_dim
        :param batch_virus_ft:     tensor    batch, virus_dim
        :return:
        '''
        batch_size = batch_antibody_onehot_ft.size()[0]

        batch_virus_onehot_ft = batch_virus_onehot_ft.unsqueeze(1)
        batch_antibody_onehot_ft = batch_antibody_onehot_ft.unsqueeze(1)

        virus_ft = self.res_net(batch_virus_onehot_ft)
        antibody_ft = self.res_net(batch_antibody_onehot_ft)

        virus_ft = F.max_pool2d(virus_ft, kernel_size=[self.max_virus_len, 1]).view(batch_size, -1)
        antibody_ft = F.max_pool2d(antibody_ft, kernel_size=[self.max_antibody_len, 1]).view(batch_size, -1)

        pair_ft = torch.cat([virus_ft, antibody_ft], dim=-1)

        pair_ft = self.out_linear1(pair_ft)
        pair_ft = self.activation(pair_ft)
        pred = self.out_linear2(pair_ft)

        return pred
