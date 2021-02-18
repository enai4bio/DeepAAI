#!/usr/bin/env
# coding:utf-8

"""
Created on 2020/11/9 下午4:18

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import pandas as pd
import torch.nn as nn
from dataset.abs_dataset_cls import AbsDataset
from dataset.abs_dataset_pdb import PDBDataset
from models.deep_aai_kmer_embedding_cls import DeepAAIKmerEmbeddingCls
from metrics.evaluate import evaluate_classification
from sklearn.metrics import confusion_matrix
import numpy as np
import random
import torch
import math
import os.path as osp

current_path = osp.dirname(osp.realpath(__file__))

class Trainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict

        self.setup_seed(self.param_dict['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hiv_dataset = AbsDataset(max_antibody_len=self.param_dict['max_antibody_len'],
                                  max_virus_len=self.param_dict['max_virus_len'],
                                  train_split_param=self.param_dict['hot_data_split'],
                                  label_type=self.param_dict['label_type'],
                                  kmer_min_df=self.param_dict['kmer_min_df'],
                                  reprocess=False)
        self.param_dict.update(self.hiv_dataset.dataset_info)
        self.hiv_dataset.to_tensor(self.device)
        self.pdb_dataset = PDBDataset(
            max_antibody_len='auto',
            max_virus_len='auto',
            split_param=self.param_dict['transfer_split_param'],
            kmer_min_df=self.param_dict['kmer_min_df'],
            select_type=self.param_dict['transfer_type'],
            generate_negative_pair=True
        )
        self.pdb_dataset.to_tensor(self.device)
        self.file_name = __file__.split('/')[-1].replace('.py', '')
        self.trainer_info = '{}_seed={}_batch={}_transfer_type={}'.format(
            self.file_name, self.param_dict['seed'], self.param_dict['batch_size'], self.param_dict['transfer_type'])
        self.loss_op = nn.BCELoss()
        self.build_model()

        # load param
        model_file_path = osp.join(current_path, '..', 'save_model_param_pred', 'deep_aai_k+e',
                                   'deep_aai_kmer_embedding_cls_seed=2_param.pkl')
        self.model.load_state_dict(torch.load(model_file_path))
        # print('load_param ', model_file_path)
        requires_grad_params = ['pred_linear', 'local_linear', 'global_linear']
        for k, v in self.model.named_parameters():
            tag = False
            for param_name in requires_grad_params:
                if param_name in k :
                    tag = True
                    break
            v.requires_grad = tag  # freeze parameters

    def build_model(self):
        self.model = DeepAAIKmerEmbeddingCls(**self.param_dict).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'])
        self.best_res = None
        self.min_dif = -1e10

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def print_res(self, res_list, epoch):
        train_acc, seen_valid_acc,\
        train_p, seen_valid_p,\
        train_r, seen_valid_r,\
        train_f1, seen_valid_f1,\
        train_auc, seen_valid_auc,\
        train_mcc, seen_valid_mcc, = res_list

        msg_log = 'Epoch: {:03d}, ' \
                  'ACC: Train {:.4f}, Val: {:.4f}, ' \
                  'P: Train {:.4f}, Val: {:.4f}, ' \
                  'R: Train {:.4f}, Val: {:.4f},  ' \
                  'F1: Train {:.4f}, Val: {:.4f},  ' \
                  'AUC: Train {:.4f}, Val: {:.4f}, ' \
                  'MCC: Train {:.4f}, Val: {:.4f},   ' \
            .format(epoch,
                    train_acc, seen_valid_acc,
                    train_p, seen_valid_p, \
                    train_r, seen_valid_r, \
                    train_f1, seen_valid_f1, \
                    train_auc, seen_valid_auc, \
                    train_mcc, seen_valid_mcc,
                    )
        print(msg_log)


    def transfer_learning(self, display=True):
        for epoch in range(1, self.param_dict['epoch_num'] + 1):
            # train

            train_pred, train_label, train_loss = self.transfer_iteration(epoch, self.pdb_dataset.train_index,
                                                                 is_training=True, shuffle=True)
            train_acc, train_p, train_r, train_f1, train_auc, train_mcc = \
                evaluate_classification(predict_proba=train_pred, label=train_label)

            valid_pred, valid_label, valid_loss = self.transfer_iteration(epoch, self.pdb_dataset.valid_index,
                                                                 is_training=False, shuffle=False)
            seen_valid_acc, seen_valid_p, seen_valid_r, seen_valid_f1, seen_valid_auc, seen_valid_mcc = \
                evaluate_classification(predict_proba=valid_pred, label=valid_label)

            res_list = [
                train_acc, seen_valid_acc,
                train_p, seen_valid_p,
                train_r, seen_valid_r,
                train_f1, seen_valid_f1,
                train_auc, seen_valid_auc,
                train_mcc, seen_valid_mcc,
            ]
            if display:
                self.print_res(res_list, epoch)

        # same_model_param_path = osp.join(current_path, save_model_data_dir, self.trainer_info + '_param.pkl')
        # torch.save(self.model.state_dict(), same_model_param_path)

        # generate_confusion_matrix

        valid_pred = valid_pred > 0.5
        # print(valid_label, valid_pred)
        mat = confusion_matrix(y_pred=valid_pred, y_true=valid_label, labels=[0, 1])
        print(mat)
        if self.param_dict['save_res']:
            df = pd.DataFrame(mat, columns=['n', 'p'], index=['n', 'y'])
            df.to_csv(
                osp.join(current_path, 'save_model_param_pred', 'deep_aai_k+e', 'transfer_result',
                         self.trainer_info+'_confusion_matrix.csv')
            )
            # for idx in range(valid_label.shape[0]):
            #     print(valid_label[idx], valid_pred[idx])

    def transfer_iteration(self, epoch, pair_idx, is_training=True, shuffle=True):
        # all_node in graph
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

        hiv_antibody_kmer_ft = self.hiv_dataset.protein_ft_dict['antibody_kmer_whole']
        hiv_virus_kmer_ft = self.hiv_dataset.protein_ft_dict['virus_kmer_whole']
        pdb_antibody_kmer_ft = self.pdb_dataset.protein_ft_dict['antibody_kmer_whole']
        pdb_virus_kmer_ft = self.pdb_dataset.protein_ft_dict['virus_kmer_whole']

        all_antibody_kmer_ft = torch.cat([hiv_antibody_kmer_ft, pdb_antibody_kmer_ft], dim=0)
        all_virus_kmer_ft = torch.cat([hiv_virus_kmer_ft, pdb_virus_kmer_ft], dim=0)

        pdb_antibody_idx_to_graph_idx = np.arange(len(self.pdb_dataset.raw_all_antibody_set)) + \
                                        len(self.hiv_dataset.raw_all_antibody_set)
        pdb_virus_idx_to_graph_idx = np.arange(len(self.pdb_dataset.raw_all_virus_set)) + \
                                        len(self.hiv_dataset.raw_all_virus_set)

        for i in range(math.ceil(pair_num / self.param_dict['batch_size'])):
            right_bound = min((i + 1) * self.param_dict['batch_size'], pair_num + 1)
            batch_idx = range_idx[i * self.param_dict['batch_size']: right_bound]
            shuffled_batch_idx = pair_idx[batch_idx]

            batch_antibody_idx = self.pdb_dataset.antibody_index_in_pair[shuffled_batch_idx]
            batch_virus_idx = self.pdb_dataset.virus_index_in_pair[shuffled_batch_idx]
            batch_label = self.pdb_dataset.all_label_mat[shuffled_batch_idx]
            batch_tensor_label = torch.FloatTensor(batch_label).to(self.device)

            batch_antibody_amino_ft = self.pdb_dataset.protein_ft_dict['antibody_amino_num'][batch_antibody_idx]
            batch_virus_amino_ft = self.pdb_dataset.protein_ft_dict['virus_amino_num'][batch_virus_idx]

            # index_remap:  raw_index -> graph_index
            # batch_antibody_idx -> batch_antibody_node_idx_in_graph
            # batch_virus_idx -> batch_virus_node_idx_in_graph
            batch_antibody_node_idx_in_graph = pdb_antibody_idx_to_graph_idx[batch_antibody_idx]
            batch_virus_node_idx_in_graph = pdb_virus_idx_to_graph_idx[batch_virus_idx]

            ft_dict = {
                'antibody_graph_node_kmer_ft': all_antibody_kmer_ft,
                'virus_graph_node_kmer_ft': all_virus_kmer_ft,
                'antibody_amino_ft': batch_antibody_amino_ft,
                'virus_amino_ft': batch_virus_amino_ft,
                'antibody_idx': batch_antibody_node_idx_in_graph,
                'virus_idx': batch_virus_node_idx_in_graph
            }

            pred, antibody_adj, virus_adj = self.model(**ft_dict)
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


if __name__ == '__main__':
    for seed in range(1, 6):
        print('seed = ', seed)
        param_dict = {
            'hot_data_split': [0.9, 0.05, 0.05],
            # 'seed': 17,
            'seed': seed,
            'kmer_min_df': 0.1,
            'label_type': 'label_10',
            'batch_size': 2,
            'epoch_num': 60,
            'h_dim': 512,
            'dropout_num': 0.4,
            'lr': 1e-5,
            'amino_embedding_dim': 7,
            'kernel_cfg': [7, 9, 11],
            'channel_cfg': [256, 256, 256],
            'dilation_cfg': [1, 1, 1],
            'adj_loss_coef': 5e-4,
            'param_l2_coef': 5e-4,
            'add_res': True,
            'add_bn': False,
            'max_antibody_len': 344,
            'max_virus_len': 912,
            'transfer_type': ['ebola'],
            'transfer_split_param': [0.5, 0.5],
            'save_res': False
        }
        trainer = Trainer(**param_dict)
        trainer.transfer_learning()