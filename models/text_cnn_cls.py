#!/usr/bin/env
# coding:utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn


class TextInception(nn.Module):
    def __init__(self, in_channel, kernel_width):
        super(TextInception, self).__init__()
        self.kernel_width = kernel_width

        self.conv1 = nn.Conv2d(in_channel, 1, kernel_size=(5, kernel_width), padding=(2, kernel_width//2))
        self.conv2 = nn.Conv2d(in_channel, 1, kernel_size=(7, kernel_width), padding=(3, kernel_width//2))
        self.conv3 = nn.Conv2d(in_channel, 1, kernel_size=(9, kernel_width), padding=(4, kernel_width//2))

    def forward(self, protein_ft):
        '''
        :param protein_ft: batch*len*amino_dim
        :return:
        '''
        batch_size = protein_ft.size()[0]
        protein_ft = protein_ft.transpose(1, 2)
        protein_ft = protein_ft.unsqueeze(1)

        conv1_ft = F.relu(self.conv1(protein_ft))
        conv2_ft = F.relu(self.conv2(protein_ft))
        conv3_ft = F.relu(self.conv3(protein_ft))

        ft1, _ = torch.max(conv1_ft, dim=-1)
        ft2, _ = torch.max(conv2_ft, dim=-1)
        ft3, _ = torch.max(conv3_ft, dim=-1)
        cat_ft = torch.cat([ft1.view(batch_size, -1), ft2.view(batch_size, -1), ft3.view(batch_size, -1)], dim=1)
        return cat_ft


class TextCNNCls(nn.Module):
    def __init__(self, amino_ft_dim, max_antibody_len, max_virus_len,
                 h_dim=512,
                 dropout=0.1,
                 ):
        super(TextCNNCls, self).__init__()
        self.amino_ft_dim = amino_ft_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.inception_out_channel = 3
        self.max_antibody_len = max_antibody_len
        self.max_virus_len = max_virus_len

        self.text_inception = TextInception(in_channel=1, kernel_width=amino_ft_dim)
        self.text_inception2 = TextInception(in_channel=1, kernel_width=amino_ft_dim)

        self.out_linear1 = nn.Linear(self.amino_ft_dim * 6, self.h_dim)
        self.out_linear2 = nn.Linear(self.h_dim, 1)
        self.activation = nn.ELU()

    def forward(self, batch_antibody_ft, batch_virus_ft):
        '''
        :param batch_antibody_ft:   tensor    batch, len, amino_ft_dim
        :param batch_virus_ft:     tensor    batch, amino_ft_dim
        :return:
        '''

        batch_size = batch_antibody_ft.size()[0]
        antibody_ft = self.text_inception(batch_antibody_ft).view(batch_size, -1)
        virus_ft = self.text_inception2(batch_virus_ft).view(batch_size, -1)

        pair_ft = torch.cat([virus_ft, antibody_ft], dim=-1).view(batch_size, -1)
        pair_ft = self.activation(pair_ft)

        pair_ft = self.out_linear1(pair_ft)
        pair_ft = self.activation(pair_ft)
        pred = self.out_linear2(pair_ft)
        return torch.sigmoid(pred)
