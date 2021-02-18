#!/usr/bin/env
# coding:utf-8

"""
Created on 2021/1/26 下午3:33

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
import torch.nn.functional as F
import torch.nn as nn
from models.layers.gcn_conv_input_mat import GCNConv

import torch.nn.functional as F
import torch.nn as nn
from models.layers.gcn_conv_input_mat import GCNConv


class BasicBlock1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride, add_bn=True, add_res=False, dilation=1):
        super(BasicBlock1D, self).__init__()
        self.add_bn = add_bn
        self.add_res = add_res
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        if self.add_bn:
            self.bn = nn.BatchNorm1d(out_channel)
        if (kernel_size - 1) * dilation // 2 != padding or stride != 1:
            exit('input dim != out dim')

    def forward(self, x):
        if self.add_res:
            residual = x
        out = self.conv(x)
        out = F.relu(out)
        if self.add_bn:
            out = self.bn(out)
        if self.add_res and self.in_channel == self.out_channel:
            out += residual
        return out


class DeepAAIKmerPssmCls(nn.Module):
    def __init__(self, **param_dict):
        super(DeepAAIKmerPssmCls, self).__init__()
        self.param_dict = param_dict
        self.kmer_dim = param_dict['kmer_dim']
        self.h_dim = param_dict['h_dim']
        self.dropout = param_dict['dropout_num']
        self.add_bn = param_dict['add_bn']
        self.add_res = param_dict['add_res']
        self.amino_embedding_dim = param_dict['amino_embedding_dim']
        self.kernel_cfg = param_dict['kernel_cfg']
        self.channel_cfg = param_dict['channel_cfg']
        self.dilation_cfg = param_dict['dilation_cfg']

        self.antibody_kmer_linear = nn.Linear(param_dict['kmer_dim'], self.h_dim)
        self.virus_kmer_linear = nn.Linear(param_dict['kmer_dim'], self.h_dim)

        self.antibody_pssm_linear = nn.Linear(param_dict['pssm_antibody_dim'], self.h_dim)
        self.virus_pssm_linear = nn.Linear(param_dict['pssm_virus_dim'], self.h_dim)

        self.share_linear = nn.Linear(self.h_dim * 2, self.h_dim)
        self.share_gcn1 = GCNConv(self.h_dim, self.h_dim)
        self.share_gcn2 = GCNConv(self.h_dim, self.h_dim)

        self.antibody_adj_trans = nn.Linear(self.h_dim, self.h_dim)
        self.virus_adj_trans = nn.Linear(self.h_dim, self.h_dim)

        self.cross_scale_merge = nn.Parameter(
            torch.ones(1)
        )

        self.global_linear = nn.Linear(self.h_dim * 2, self.h_dim)
        self.pred_linear = nn.Linear(self.h_dim, 1)

        self.activation = nn.ELU()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)


    def forward(self, **ft_dict):
        '''
        :param ft_dict:
                ft_dict = {
                'antibody_graph_node_ft': FloatTensor  node_num * kmer_dim
                'virus_graph_node_ft': FloatTensor  node_num * kmer_dim,
                'antibody_amino_ft': LongTensor  batch * max_antibody_len * 1
                'virus_amino_ft': LongTensor  batch * max_virus_len * 1,
                'antibody_idx': LongTensor  batch
                'virus_idx': LongTensor  batch
            }
        :return:
        '''
        device = ft_dict['antibody_graph_node_kmer_ft'].device
        antibody_graph_node_num = ft_dict['antibody_graph_node_kmer_ft'].size()[0]
        virus_graph_node_num = ft_dict['virus_graph_node_kmer_ft'].size()[0]
        antibody_res_mat = torch.zeros(antibody_graph_node_num, self.h_dim).to(device)
        virus_res_mat = torch.zeros(virus_graph_node_num, self.h_dim).to(device)

        antibody_node_kmer_ft = self.antibody_kmer_linear(ft_dict['antibody_graph_node_kmer_ft'])
        antibody_node_pssm_ft = self.antibody_pssm_linear(ft_dict['antibody_graph_node_pssm_ft'])
        antibody_node_ft = torch.cat([antibody_node_kmer_ft, antibody_node_pssm_ft], dim=-1)

        antibody_node_ft = self.activation(antibody_node_ft)
        antibody_node_ft = F.dropout(antibody_node_ft, p=self.dropout, training=self.training)

        # share
        antibody_node_ft = self.share_linear(antibody_node_ft)
        antibody_res_mat = antibody_res_mat + antibody_node_ft
        antibody_node_ft = self.activation(antibody_node_ft)
        antibody_node_ft = F.dropout(antibody_node_ft, p=self.dropout, training=self.training)

        # generate antibody adj
        antibody_trans_ft = self.antibody_adj_trans(antibody_node_ft)
        antibody_trans_ft = torch.tanh(antibody_trans_ft)
        w = torch.norm(antibody_trans_ft, p=2, dim=-1).view(-1, 1)
        w_mat = w * w.t()
        antibody_adj = torch.mm(antibody_trans_ft, antibody_trans_ft.t()) / w_mat

        antibody_node_ft = self.share_gcn1(antibody_node_ft, antibody_adj)
        antibody_res_mat = antibody_res_mat + antibody_node_ft

        antibody_node_ft = self.activation(antibody_res_mat)  # add
        antibody_node_ft = F.dropout(antibody_node_ft, p=self.dropout, training=self.training)
        antibody_node_ft = self.share_gcn2(antibody_node_ft, antibody_adj)
        antibody_res_mat = antibody_res_mat + antibody_node_ft

        # virus
        virus_node_kmer_ft = self.virus_kmer_linear(ft_dict['virus_graph_node_kmer_ft'])
        virus_node_pssm_ft = self.virus_pssm_linear(ft_dict['virus_graph_node_pssm_ft'])
        virus_node_ft = torch.cat([virus_node_kmer_ft, virus_node_pssm_ft], dim=-1)
        # virus_node_ft = virus_node_kmer_ft
        virus_node_ft = self.activation(virus_node_ft)
        virus_node_ft = F.dropout(virus_node_ft, p=self.dropout, training=self.training)

        # share
        virus_node_ft = self.share_linear(virus_node_ft)
        virus_res_mat = virus_res_mat + virus_node_ft
        virus_node_ft = self.activation(virus_node_ft)
        virus_node_ft = F.dropout(virus_node_ft, p=self.dropout, training=self.training)

        # generate antibody adj
        virus_trans_ft = self.virus_adj_trans(virus_node_ft)
        virus_trans_ft = torch.tanh(virus_trans_ft)
        w = torch.norm(virus_trans_ft, p=2, dim=-1).view(-1, 1)
        w_mat = w * w.t()
        virus_adj = torch.mm(virus_trans_ft, virus_trans_ft.t()) / w_mat

        virus_node_ft = self.share_gcn1(virus_node_ft, virus_adj)
        virus_res_mat = virus_res_mat + virus_node_ft

        virus_node_ft = self.activation(virus_res_mat)  # add
        virus_node_ft = F.dropout(virus_node_ft, p=self.dropout, training=self.training)
        virus_node_ft = self.share_gcn2(virus_node_ft, virus_adj)
        virus_res_mat = virus_res_mat + virus_node_ft

        antibody_res_mat = self.activation(antibody_res_mat)
        virus_res_mat = self.activation(virus_res_mat)

        antibody_res_mat = antibody_res_mat[ft_dict['antibody_idx']]
        virus_res_mat = virus_res_mat[ft_dict['virus_idx']]

        # cross
        global_pair_ft = torch.cat([virus_res_mat, antibody_res_mat], dim=1)
        global_pair_ft = self.activation(global_pair_ft)
        global_pair_ft = F.dropout(global_pair_ft, p=self.dropout, training=self.training)
        global_pair_ft = self.global_linear(global_pair_ft)

        pair_ft = global_pair_ft
        pair_ft = self.activation(pair_ft)
        pair_ft = F.dropout(pair_ft, p=self.dropout, training=self.training)

        pred = self.pred_linear(pair_ft)
        pred = torch.sigmoid(pred)
        return pred, antibody_adj, virus_adj

