#!/usr/bin/env
# coding:utf-8

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

class ParapredBase(nn.Module):
    def __init__(self, ft_dim):
        super(ParapredBase, self).__init__()
        self.lstm_hidden_size = 128
        self.conv1 = nn.Conv1d(ft_dim, 32, 3, padding=1)  # antibody first a trous convolutional layer
        self.conv2 = nn.Conv1d(32, 32, 3, padding=2, dilation=2)  #antibody second a trous convolutional layer
        self.conv3 = nn.Conv1d(32, 64, 3, padding=4, dilation=4)  #antibody third a trous convolutional layer

        self.bn1 = nn.BatchNorm1d(32)  # batch normalisation after the first convolutional layer for antibody
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)

        self.dp_cnn = nn.Dropout(.15)
        self.lstm = nn.LSTM(64, 128, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.dp_lstm = nn.Dropout(.3)
        self.activation = nn.ReLU()

    def forward(self, protein_ft, protein_len):
        self.lstm.flatten_parameters()

        batch_size = protein_ft.size()[0]
        protein_ft = protein_ft.transpose(1, 2)  # 维度转换=> batchSize × feaSize × seqLen

        # generate mask mat
        mask_mat = torch.zeros(batch_size, 1, protein_ft.size()[-1]).to(protein_ft.device)
        for idx in range(batch_size):
            mask_mat[idx, :, :protein_len[idx]] = 1

        # antibody ft extraction
        protein_ft = torch.mul(protein_ft, mask_mat)
        protein_ft = self.conv1(protein_ft)
        protein_ft = torch.mul(protein_ft, mask_mat)
        protein_ft = self.bn1(protein_ft)
        protein_ft = self.activation(protein_ft)

        protein_ft = self.conv2(protein_ft)
        protein_ft = torch.mul(protein_ft, mask_mat)
        protein_ft = self.bn2(protein_ft)
        protein_ft = self.activation(protein_ft)

        protein_ft = self.conv3(protein_ft)
        protein_ft = torch.mul(protein_ft, mask_mat)
        protein_ft = self.bn3(protein_ft)
        protein_ft = self.activation(protein_ft)

        protein_ft = protein_ft.transpose(1, 2)
        h0 = torch.randn(2, batch_size, self.lstm_hidden_size).to(protein_ft.device)
        c0 = torch.randn(2, batch_size, self.lstm_hidden_size).to(protein_ft.device)
        protein_ft, (hn, cn) = self.lstm(protein_ft, (h0, c0))  # lstm
        protein_ft = F.dropout(protein_ft, p=0.3, training=self.training)
        out, d = protein_ft.max(dim=-1)
        return out


class ParapredReg(nn.Module):
    def __init__(self, h_dim, max_antibody_len, max_virus_len,
                 amino_ft_dim=22,
                 ):
        super(ParapredReg, self).__init__()
        self.max_virus_len = max_virus_len
        self.max_antibody_len = max_antibody_len
        self.h_dim = h_dim
        self.amino_ft_dim = amino_ft_dim

        self.antibody_parapred = ParapredBase(amino_ft_dim)
        self.virus_parapred = ParapredBase(amino_ft_dim)

        self.final_linear1 = nn.Linear(self.max_virus_len + self.max_antibody_len, self.h_dim)
        self.final_linear2 = nn.Linear(self.h_dim, 1)
        self.activation = nn.ELU()

    def forward(self, batch_antibody_ft, batch_virus_ft, batch_antibody_len, batch_virus_len):
        '''
        :param batch_antibody_ft:   tensor    batch, max_antibody_len, amino_ft_dim
        :param batch_virus_ft:     tensor    batch, max_virus_len, amino_ft_dim
        :param batch_antibody_len:   np array      batch
        :param batch_virus_len:     np array      batch
        :return:
        '''
        assert batch_antibody_ft.size()[0] == batch_virus_ft.size()[0]

        antibody_ft = self.antibody_parapred(batch_antibody_ft, batch_antibody_len)
        virus_ft = self.virus_parapred(batch_virus_ft, batch_virus_len)

        pair_ft = torch.cat((antibody_ft, virus_ft), dim=-1)

        pair_ft = self.final_linear1(pair_ft)
        pair_ft = self.activation(pair_ft)
        pred = self.final_linear2(pair_ft)
        return pred
