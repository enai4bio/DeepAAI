#!/usr/bin/env
# coding:utf-8

import torch.nn.functional as F
import torch.nn as nn
import torch


class PIPRReg(nn.Module):
    def __init__(self, protein_ft_one_hot_dim):
        super(PIPRReg, self).__init__()
        self.protein_ft_dim = protein_ft_one_hot_dim
        self.hidden_num = 50
        self.kernel_size = 3
        self.pool_size = 3
        self.conv_layer_num = 2

        self.conv1d_layer_list = nn.ModuleList()
        for idx in range(self.conv_layer_num):
            in_channels = self.hidden_num * 2 + self.hidden_num // self.pool_size
            if idx == 0:
                in_channels = self.protein_ft_dim
            self.conv1d_layer_list.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=self.hidden_num,
                          kernel_size=self.kernel_size,
                          padding=0)
            )

        self.gru_list = nn.ModuleList()
        for idx in range(self.conv_layer_num - 1):
            self.gru_list.append(
                nn.GRU(input_size=self.hidden_num // self.pool_size, hidden_size=self.hidden_num, batch_first=True,
                       bidirectional=True)
            )

        self.linear_pred = nn.Linear(self.hidden_num, 1)

    def block_cnn_rnn(self, cnn_layer, rnn_layer, ft_mat, kernel_wide=2):
        '''
        :param cnn_layer:
        :param rnn_layer:
        :param ft_mat:     (batch, features, steps)
        :param kernel_wide:
        :return:
        '''

        # print("in ", ft_mat.size())
        ft_mat = ft_mat.transpose(1, 2)
        ft_mat = cnn_layer(ft_mat)
        ft_mat = ft_mat.transpose(1, 2)

        ft_mat = F.max_pool1d(ft_mat, kernel_wide)
        h0 = torch.randn(2, ft_mat.size()[0], self.hidden_num).to(ft_mat.device)
        output, hn = rnn_layer(ft_mat, h0)
        output = torch.cat([ft_mat, output], dim=-1)
        return output

    def forward(self, antibody_ft, virus_ft):
        for gru_layer in self.gru_list:
            gru_layer.flatten_parameters()
        # batch  * seq_len * feature
        # print('1 antibody_ft = ', antibody_ft[:, :, 1])
        for idx in range(self.conv_layer_num - 1):
            antibody_ft = self.block_cnn_rnn(
                cnn_layer=self.conv1d_layer_list[idx],
                rnn_layer=self.gru_list[idx],
                ft_mat=antibody_ft,
                kernel_wide=self.pool_size
            )
        antibody_ft = self.conv1d_layer_list[-1](
            antibody_ft.transpose(1, 2))  # (batch, features, steps)

        antibody_ft = F.max_pool2d(
            antibody_ft, kernel_size=(1, antibody_ft.size()[-1])).squeeze()  # (batch, features)

        for idx in range(self.conv_layer_num - 1):
            virus_ft = self.block_cnn_rnn(
                cnn_layer=self.conv1d_layer_list[idx],
                rnn_layer=self.gru_list[idx],
                ft_mat=virus_ft,
                kernel_wide=self.pool_size
            )
        virus_ft = self.conv1d_layer_list[-1](
            virus_ft.transpose(1, 2))  # (batch, features, steps)
        virus_ft = F.max_pool2d(
            virus_ft, kernel_size=(1, virus_ft.size()[-1])).squeeze()  # (batch, features)

        pair_ft = antibody_ft + virus_ft
        pred = self.linear_pred(pair_ft)

        return torch.sigmoid(pred)