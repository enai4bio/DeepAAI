#!/usr/bin/env
# coding:utf-8

import torch
import torch.nn as nn
from models.layers.attention_layer import Att2One


class BiLSTMAttentionCls(nn.Module):
    def __init__(self, **param_dict):
        super(BiLSTMAttentionCls, self).__init__()
        self.h_dim = param_dict['h_dim']
        self.amino_ft_dim = param_dict['amino_type_num']
        self.dropout_num = param_dict['dropout_num']
        self.amino_embedding_dim = param_dict['amino_embedding_dim']

        self.amino_embedding_layer = nn.Embedding(param_dict['amino_type_num'], self.amino_embedding_dim)
        self.share_bi_lstm_layer = nn.LSTM(
            input_size=self.amino_embedding_dim,
            hidden_size=self.h_dim // 2,
            num_layers=param_dict['bilstm_num_layers'],
            dropout=self.dropout_num,
            batch_first=True,
            bidirectional=True
        )

        self.share_att_layer = Att2One(self.h_dim, self.h_dim)
        self.dropout_layer = nn.Dropout(self.dropout_num)
        self.linear_merge = nn.Linear(self.h_dim * 2, self.h_dim)
        self.linear_pred = nn.Linear(self.h_dim, 1)
        self.activation = nn.ELU()

    def forward(self,
                batch_antibody_amino_ft,
                batch_virus_amino_ft
                ):
        self.share_bi_lstm_layer.flatten_parameters()
        batch_size = batch_antibody_amino_ft.size()[0]
        deivce = batch_antibody_amino_ft.device

        virus_ft = self.amino_embedding_layer(batch_virus_amino_ft)
        antibody_ft = self.amino_embedding_layer(batch_antibody_amino_ft)

        h_0 = torch.randn(2 * self.share_bi_lstm_layer.num_layers, batch_size, self.share_bi_lstm_layer.hidden_size).to(deivce)
        c_0 = torch.randn(2 * self.share_bi_lstm_layer.num_layers, batch_size, self.share_bi_lstm_layer.hidden_size).to(deivce)
        virus_ft, (hn, cn) = self.share_bi_lstm_layer(virus_ft, (h_0, c_0))

        h_0 = torch.randn(2 * self.share_bi_lstm_layer.num_layers, batch_size, self.share_bi_lstm_layer.hidden_size).to(deivce)
        c_0 = torch.randn(2 * self.share_bi_lstm_layer.num_layers, batch_size, self.share_bi_lstm_layer.hidden_size).to(deivce)
        antibody_ft, (hn, cn) = self.share_bi_lstm_layer(antibody_ft, (h_0, c_0))

        antibody_ft = self.activation(antibody_ft)
        virus_ft = self.activation(virus_ft)
        antibody_ft = self.dropout_layer(antibody_ft)
        virus_ft = self.dropout_layer(virus_ft)

        antibody_ft = self.share_att_layer(antibody_ft)
        virus_ft = self.share_att_layer(virus_ft)

        pair_ft = torch.cat([virus_ft, antibody_ft], dim=-1)
        pair_ft = self.activation(pair_ft)
        pair_ft = self.dropout_layer(pair_ft)

        pair_ft = self.linear_merge(pair_ft)
        pair_ft = self.activation(pair_ft)
        pair_ft = self.dropout_layer(pair_ft)
        pred = self.linear_pred(pair_ft)

        return torch.sigmoid(pred)