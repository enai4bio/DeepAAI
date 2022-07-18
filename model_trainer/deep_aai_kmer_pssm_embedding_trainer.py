#!/usr/bin/env
# coding:utf-8

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
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


class Model(nn.Module):
    def __init__(self, **param_dict):
        super(Model, self).__init__()
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

        # local
        self.amino_embedding_layer = nn.Embedding(param_dict['amino_type_num'], self.amino_embedding_dim)
        self.channel_cfg.insert(0, self.amino_embedding_dim)
        cnn_layer_list = []
        for idx in range(len(self.kernel_cfg)):
            cnn_layer_list.append(
                BasicBlock1D(in_channel=self.channel_cfg[idx],
                             out_channel=self.channel_cfg[idx+1],
                             kernel_size=self.kernel_cfg[idx],
                             padding=(self.kernel_cfg[idx] - 1)*self.dilation_cfg[idx]//2,
                             stride=1,
                             add_bn=self.add_bn,
                             add_res=self.add_res,
                             dilation=self.dilation_cfg[idx]
                             )
            )
        self.conv_sq = nn.Sequential(*cnn_layer_list)
        # self.local_linear = nn.Linear(self.channel_cfg[-1], self.h_dim)
        self.local_linear = nn.Linear(self.channel_cfg[-1] * 2, self.h_dim)
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
        # virus_adj = eye_adj

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
        # global_pair_ft = virus_res_mat + antibody_res_mat
        global_pair_ft = torch.cat([virus_res_mat, antibody_res_mat], dim=1)
        global_pair_ft = self.activation(global_pair_ft)
        global_pair_ft = F.dropout(global_pair_ft, p=self.dropout, training=self.training)
        global_pair_ft = self.global_linear(global_pair_ft)

        antibody_amino_ft = self.amino_embedding_layer(ft_dict['antibody_amino_ft']).transpose(1, 2)
        virus_amino_ft = self.amino_embedding_layer(ft_dict['virus_amino_ft']).transpose(1, 2)

        antibody_amino_ft = self.conv_sq(antibody_amino_ft)
        virus_amino_ft = self.conv_sq(virus_amino_ft)

        # print(antibody_amino_ft.size(), virus_amino_ft.size())
        antibody_amino_ft, _ = torch.max(antibody_amino_ft, dim=-1)
        virus_amino_ft, _ = torch.max(virus_amino_ft, dim=-1)

        # local_pair_ft = antibody_amino_ft + virus_amino_ft + (antibody_amino_ft * virus_amino_ft) * self.cross_scale_local
        local_pair_ft = torch.cat([antibody_amino_ft, virus_amino_ft], dim=-1)
        local_pair_ft = self.activation(local_pair_ft)
        local_pair_ft = self.local_linear(local_pair_ft)

        # print('global_pair_ft = ', global_pair_ft)
        # print('local_pair_ft = ', local_pair_ft)

        pair_ft = global_pair_ft + local_pair_ft + (global_pair_ft * local_pair_ft) * self.cross_scale_merge
        pair_ft = self.activation(pair_ft)
        pair_ft = F.dropout(pair_ft, p=self.dropout, training=self.training)

        pred = self.pred_linear(pair_ft)
        pred = torch.sigmoid(pred)
        # print('pred = ', pred)
        return pred, antibody_adj, virus_adj, pair_ft


from dataset.abs_dataset_cls import AbsDataset
from metrics.evaluate import evaluate_classification
import numpy as np
import random
import torch
import math
import os.path as osp
from utils.index_map import get_map_index_for_sub_arr

current_path = osp.dirname(osp.realpath(__file__))
select_model_param_dir = 'select_model_param'

class Trainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict
        self.setup_seed(self.param_dict['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = AbsDataset(max_antibody_len=self.param_dict['max_antibody_len'],
                                  max_virus_len=self.param_dict['max_virus_len'],
                                  train_split_param=self.param_dict['hot_data_split'],
                                  label_type=self.param_dict['label_type'],
                                  kmer_min_df=self.param_dict['kmer_min_df'],
                                  reprocess=True)
        self.param_dict.update(self.dataset.dataset_info)
        self.dataset.to_tensor(self.device)
        self.file_name = __file__.split('/')[-1].replace('.py', '')
        self.trainer_info = '{}_seed={}_batch={}'.format(self.file_name, self.param_dict['seed'], self.param_dict['batch_size'])
        self.loss_op = nn.BCELoss()
        self.build_model()

    def build_model(self):
        self.model = Model(**self.param_dict).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'])
        self.best_res = None
        self.min_dif = -1e10

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def iteration(self, epoch, pair_idx, antibody_graph_node_idx, virus_graph_node_idx, is_training=True, shuffle=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        pair_num = pair_idx.shape[0]
        if shuffle is True:
            range_idx = np.random.permutation(pair_num)
        else:
            range_idx = np.arange(0, pair_num)

        all_pred = []
        all_label = []
        all_loss = []

        antibody_graph_node_kmer_ft = self.dataset.protein_ft_dict['antibody_kmer_whole'][antibody_graph_node_idx]
        virus_graph_node_kmer_ft = self.dataset.protein_ft_dict['virus_kmer_whole'][virus_graph_node_idx]
        antibody_graph_node_pssm_ft = self.dataset.protein_ft_dict['antibody_pssm'][antibody_graph_node_idx]
        virus_graph_node_pssm_ft = self.dataset.protein_ft_dict['virus_pssm'][virus_graph_node_idx]
        antibody_graph_map_arr = get_map_index_for_sub_arr(
            antibody_graph_node_idx, np.arange(0, len(self.dataset.raw_all_antibody_set)))
        virus_graph_map_arr = get_map_index_for_sub_arr(
            virus_graph_node_idx, np.arange(0, len(self.dataset.raw_all_virus_set)))

        for i in range(math.ceil(pair_num/self.param_dict['batch_size'])):
            right_bound = min((i + 1)*self.param_dict['batch_size'], pair_num + 1)
            batch_idx = range_idx[i * self.param_dict['batch_size']: right_bound]
            shuffled_batch_idx = pair_idx[batch_idx]

            batch_antibody_idx = self.dataset.antibody_index_in_pair[shuffled_batch_idx]
            batch_virus_idx = self.dataset.virus_index_in_pair[shuffled_batch_idx]
            batch_label = self.dataset.all_label_mat[shuffled_batch_idx]
            batch_tensor_label = torch.FloatTensor(batch_label).to(self.device)

            batch_antibody_amino_ft = self.dataset.protein_ft_dict['antibody_amino_num'][batch_antibody_idx]
            batch_virus_amino_ft = self.dataset.protein_ft_dict['virus_amino_num'][batch_virus_idx]

            # index_remap:  raw_index -> graph_index
            # batch_antibody_idx -> batch_antibody_node_idx_in_graph
            # batch_virus_idx -> batch_virus_node_idx_in_graph
            batch_antibody_node_idx_in_graph = antibody_graph_map_arr[batch_antibody_idx]
            batch_virus_node_idx_in_graph = virus_graph_map_arr[batch_virus_idx]

            ft_dict = {
                'antibody_graph_node_kmer_ft': antibody_graph_node_kmer_ft,
                'virus_graph_node_kmer_ft': virus_graph_node_kmer_ft,
                'antibody_graph_node_pssm_ft': antibody_graph_node_pssm_ft,
                'virus_graph_node_pssm_ft': virus_graph_node_pssm_ft,
                'antibody_amino_ft': batch_antibody_amino_ft,
                'virus_amino_ft': batch_virus_amino_ft,
                'antibody_idx': batch_antibody_node_idx_in_graph,
                'virus_idx': batch_virus_node_idx_in_graph
            }

            pred, antibody_adj, virus_adj, pair_ft = self.model(**ft_dict)
            pred = pred.view(-1)
            if is_training:
                c_loss = self.loss_op(pred, batch_tensor_label)
                param_l2_loss = 0
                param_l1_loss = 0
                for name, param in self.model.named_parameters():
                    if 'bias' not in name:
                        param_l2_loss += torch.norm(param, p=2)
                        param_l1_loss += torch.norm(param, p=1)

                param_l2_loss = self.param_dict['param_l2_coef'] * param_l2_loss
                adj_l1_loss = self.param_dict['adj_loss_coef'] * torch.norm(virus_adj) + \
                              self.param_dict['adj_loss_coef'] * torch.norm(antibody_adj)
                loss = c_loss + adj_l1_loss + param_l2_loss


                all_loss.append(loss.detach().to('cpu').item())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            pred = pred.detach().to('cpu').numpy()
            all_pred = np.hstack([all_pred, pred])
            all_label = np.hstack([all_label, batch_label]).astype(np.long)

        return all_pred, all_label, all_loss

    def print_res(self, res_list, epoch):
        train_acc, seen_valid_acc, seen_test_acc, unseen_test_acc, \
        train_p, seen_valid_p, seen_test_p, unseen_test_p, \
        train_r, seen_valid_r, seen_test_r, unseen_test_r, \
        train_f1, seen_valid_f1, seen_test_f1, unseen_test_f1, \
        train_auc, seen_valid_auc, seen_test_auc, unseen_test_auc, \
        train_mcc, seen_valid_mcc, seen_test_mcc, unseen_test_mcc = res_list

        msg_log = 'Epoch: {:03d}, ' \
                  'ACC: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'P: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'R: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'F1: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'AUC: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'MCC: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f},  ' \
            .format(epoch,
                    train_acc, seen_valid_acc, seen_test_acc, unseen_test_acc,
                    train_p, seen_valid_p, seen_test_p, unseen_test_p, \
                    train_r, seen_valid_r, seen_test_r, unseen_test_r, \
                    train_f1, seen_valid_f1, seen_test_f1, unseen_test_f1, \
                    train_auc, seen_valid_auc, seen_test_auc, unseen_test_auc, \
                    train_mcc, seen_valid_mcc, seen_test_mcc, unseen_test_mcc
                    )
        print(msg_log)

    def train(self, display=True):
        for epoch in range(1, self.param_dict['epoch_num'] + 1):
            # train

            train_pred, train_label, train_loss = self.iteration(epoch, self.dataset.train_index,
                           antibody_graph_node_idx=self.dataset.known_antibody_idx,
                           virus_graph_node_idx=self.dataset.known_virus_idx,
                           is_training=True, shuffle=True)
            train_acc, train_p, train_r, train_f1, train_auc, train_mcc = \
                evaluate_classification(predict_proba=train_pred, label=train_label)

            valid_pred, valid_label, valid_loss = self.iteration(epoch, self.dataset.valid_seen_index,
                           antibody_graph_node_idx=self.dataset.known_antibody_idx,
                           virus_graph_node_idx=self.dataset.known_virus_idx,
                           is_training=False, shuffle=False)
            seen_valid_acc, seen_valid_p, seen_valid_r, seen_valid_f1, seen_valid_auc, seen_valid_mcc = \
                evaluate_classification(predict_proba=valid_pred, label=valid_label)

            seen_test_pred, seen_test_label, seen_test_loss = self.iteration(epoch, self.dataset.test_seen_index,
                           antibody_graph_node_idx=self.dataset.known_antibody_idx,
                           virus_graph_node_idx=self.dataset.known_virus_idx,
                           is_training=False, shuffle=False)
            seen_test_acc, seen_test_p, seen_test_r, seen_test_f1, seen_test_auc, seen_test_mcc = \
                evaluate_classification(predict_proba=seen_test_pred, label=seen_test_label)

            unseen_test_pred, unseen_test_label, unseen_test_loss = self.iteration(epoch, self.dataset.test_unseen_index,
                           antibody_graph_node_idx=np.hstack(
                               (self.dataset.known_antibody_idx, self.dataset.unknown_antibody_idx)),
                           virus_graph_node_idx=self.dataset.known_virus_idx,
                           is_training=False, shuffle=False)
            unseen_test_acc, unseen_test_p, unseen_test_r, unseen_test_f1, unseen_test_auc, unseen_test_mcc = \
                evaluate_classification(predict_proba=unseen_test_pred, label=unseen_test_label)

            res_list = [
                train_acc, seen_valid_acc, seen_test_acc, unseen_test_acc,
                train_p, seen_valid_p, seen_test_p, unseen_test_p,
                train_r, seen_valid_r, seen_test_r, unseen_test_r,
                train_f1, seen_valid_f1, seen_test_f1, unseen_test_f1,
                train_auc, seen_valid_auc, seen_test_auc, unseen_test_auc,
                train_mcc, seen_valid_mcc, seen_test_mcc, unseen_test_mcc
            ]

            if seen_valid_acc > self.min_dif:
                self.min_dif = seen_valid_acc
                self.best_res = res_list
                self.best_epoch = epoch

            if display:
                self.print_res(res_list, epoch)

            if epoch % 50 == 0 and epoch > 0:
                print('Best res')
                self.print_res(self.best_res, self.best_epoch)


if __name__ == '__main__':

    param_dict = {
        'hot_data_split': [0.9, 0.05, 0.05],
        'seed': 3,
        'kmer_min_df': 0.1,
        'label_type': 'label_10',
        'batch_size': 32,
        'epoch_num': 200,
        'h_dim': 512,
        'dropout_num': 0.4,
        'lr': 5e-5,
        'amino_embedding_dim': 7,
        'kernel_cfg': [7, 9, 11],
        'channel_cfg': [256, 256, 256],
        'dilation_cfg': [1, 1, 1],
        'adj_loss_coef': 1e-4,
        'param_l2_coef': 5e-4,
        'add_res': True,
        'add_bn': False,
        'max_antibody_len': 344,
        'max_virus_len': 912,
    }
    trainer = Trainer(**param_dict)
    trainer.train()
