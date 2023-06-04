#!/usr/bin/env
# coding:utf-8
import sys
import os
sys.path.append(os.getcwd())

from argparse import ArgumentParser
import torch.nn.functional as F
import torch.nn as nn
from dataset.abs_dataset_reg import AbsDataset
from metrics.evaluate import evaluate_regression_v2
from models.deep_aai_kmer_embedding_reg import DeepAAIKmerEmbeddingReg
import numpy as np
import random
import torch
import math
import os.path as osp
from utils.index_map import get_map_index_for_sub_arr
current_path = osp.dirname(osp.realpath(__file__))

class Trainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict
        self.setup_seed(self.param_dict['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = AbsDataset(max_antibody_len=self.param_dict['max_antibody_len'],
                                  max_virus_len=self.param_dict['max_virus_len'],
                                  train_split_param=self.param_dict['hot_data_split'],
                                  kmer_min_df=self.param_dict['kmer_min_df'],
                                  reprocess=False)
        self.param_dict.update(self.dataset.dataset_info)

        self.dataset.to_tensor(self.device)
        self.file_name = __file__.split('/')[-1].replace('.py', '')
        self.trainer_info = '{}_seed={}_batch={}'.format(self.file_name, self.param_dict['seed'], self.param_dict['batch_size'])
        self.save_img_path = osp.join(current_path, 'plot_curve', self.trainer_info)
        self.loss_op = nn.BCELoss()

        self.build_model()

    def build_model(self):
        self.model = DeepAAIKmerEmbeddingReg(**self.param_dict).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'])
        self.best_res = None
        self.min_dif = 1e10

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
        antibody_graph_node_ft = self.dataset.protein_ft_dict['antibody_kmer_whole'][antibody_graph_node_idx]
        virus_graph_node_ft = self.dataset.protein_ft_dict['virus_kmer_whole'][virus_graph_node_idx]

        antibody_graph_map_arr = get_map_index_for_sub_arr(
            antibody_graph_node_idx, np.arange(0, 254))
        virus_graph_map_arr = get_map_index_for_sub_arr(
            virus_graph_node_idx, np.arange(0, 940))

        for i in range(math.ceil(pair_num/self.param_dict['batch_size'])):
            right_bound = min((i + 1)*self.param_dict['batch_size'], pair_num + 1)
            batch_idx = range_idx[i * self.param_dict['batch_size']: right_bound]
            shuffled_batch_idx = pair_idx[batch_idx]

            batch_antibody_idx = self.dataset.antibody_index_in_pair[shuffled_batch_idx]
            batch_virus_idx = self.dataset.virus_index_in_pair[shuffled_batch_idx]
            batch_label = self.dataset.all_label_mat[shuffled_batch_idx]
            batch_tensor_label = torch.FloatTensor(batch_label).to(self.device)

            batch_antibody_amino_ft = self.dataset.protein_ft_dict['antibody_one_hot'][batch_antibody_idx]
            batch_virus_amino_ft = self.dataset.protein_ft_dict['virus_one_hot'][batch_virus_idx]

            # index_remap:  raw_index -> graph_index
            # batch_antibody_idx -> batch_antibody_node_idx_in_graph
            # batch_virus_idx -> batch_virus_node_idx_in_graph
            batch_antibody_node_idx_in_graph = antibody_graph_map_arr[batch_antibody_idx]
            batch_virus_node_idx_in_graph = virus_graph_map_arr[batch_virus_idx]

            ft_dict = {
                'antibody_graph_node_ft': antibody_graph_node_ft,
                'virus_graph_node_ft': virus_graph_node_ft,
                'antibody_amino_ft': batch_antibody_amino_ft,
                'virus_amino_ft': batch_virus_amino_ft,
                'antibody_idx': batch_antibody_node_idx_in_graph,
                'virus_idx': batch_virus_node_idx_in_graph
            }

            pred, antibody_adj, virus_adj = self.model(**ft_dict)
            pred = pred.view(-1)
            if is_training:
                # c_loss = self.loss_op(pred, batch_tensor_label)
                r_loss = F.mse_loss(pred, batch_tensor_label, reduction='mean')
                param_l2_loss = 0
                param_l1_loss = 0
                for name, param in self.model.named_parameters():
                    if 'bias' not in name:
                        param_l2_loss += torch.norm(param, p=2)
                        param_l1_loss += torch.norm(param, p=1)

                param_l2_loss = self.param_dict['param_l2_coef'] * param_l2_loss
                # param_l1_loss = self.param_dict['param_l1_coef'] * param_l1_loss
                adj_l1_loss = self.param_dict['adj_loss_coef'] * torch.norm(virus_adj) + \
                              self.param_dict['adj_loss_coef'] * torch.norm(antibody_adj)
                loss = r_loss + adj_l1_loss + param_l2_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            pred = pred.detach().to('cpu').numpy()
            all_pred = np.hstack([all_pred, pred])
            all_label = np.hstack([all_label, batch_label])

        # evaluate
        mse, mae, mape, r2, pccs_rho, pccs_pval, spear_rho, spear_pval = \
            evaluate_regression_v2(pred=all_pred, label=all_label)
        return mse, mae, mape, r2, pccs_rho, spear_rho

    def print_res(self, res_list, epoch):
        train_mse, valid_mse, seen_test_mse, unseen_test_mse, \
        train_mae, valid_mae, seen_test_mae, unseen_test_mae,\
        train_mape, valid_mape, seen_test_mape, unseen_test_mape,\
        train_r2, valid_r2, seen_test_r2, unseen_test_r2,\
        train_pccs_rho, valid_pccs_rho, seen_test_pccs_rho, unseen_test_pccs_rho,\
        train_spear_rho, valid_spear_rho, seen_test_spear_rho, unseen_test_spear_rho = res_list

        msg_log = 'Epoch: {:03d}, ' \
                  'MSE: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'MAE: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'MAPE: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'R_SQUARED: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'pccs_rho: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'spear_rho,: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f},  ' \
            .format(epoch,
                    train_mse, valid_mse, seen_test_mse, unseen_test_mse,
                    train_mae, valid_mae, seen_test_mae, unseen_test_mae,
                    train_mape, valid_mape, seen_test_mape, unseen_test_mape,
                    train_r2, valid_r2, seen_test_r2, unseen_test_r2,
                    train_pccs_rho, valid_pccs_rho, seen_test_pccs_rho, unseen_test_pccs_rho,
                    train_spear_rho, valid_spear_rho, seen_test_spear_rho, unseen_test_spear_rho
                    )
        print(msg_log)

    def train(self, display=True):

        for epoch in range(1, self.param_dict['epoch_num'] + 1):
            # train
            train_mse, train_mae, train_mape, train_r2, train_pccs_rho, train_spear_rho = \
                self.iteration(epoch, self.dataset.train_index,
                               antibody_graph_node_idx=self.dataset.known_antibody_idx,
                               virus_graph_node_idx=self.dataset.known_virus_idx,
                               is_training=True, shuffle=True)

            valid_mse, valid_mae, valid_mape, valid_r2, valid_pccs_rho, valid_spear_rho = \
                self.iteration(epoch, self.dataset.valid_seen_index,
                               antibody_graph_node_idx=self.dataset.known_antibody_idx,
                               virus_graph_node_idx=self.dataset.known_virus_idx,
                               is_training=False, shuffle=False)

            seen_test_mse, seen_test_mae, seen_test_mape, seen_test_r2, seen_test_pccs_rho, seen_test_spear_rho = \
                self.iteration(epoch, self.dataset.test_seen_index,
                               antibody_graph_node_idx=self.dataset.known_antibody_idx,
                               virus_graph_node_idx=self.dataset.known_virus_idx,
                               is_training=False, shuffle=False)

            unseen_test_mse, unseen_test_mae, unseen_test_mape, unseen_test_r2, unseen_test_pccs_rho, unseen_test_spear_rho = \
                self.iteration(epoch, self.dataset.test_unseen_index,
                               antibody_graph_node_idx=np.hstack((self.dataset.known_antibody_idx, self.dataset.unknown_antibody_idx)),
                               virus_graph_node_idx=self.dataset.known_virus_idx,
                               is_training=False, shuffle=False)

            res_list = [
                train_mse, valid_mse, seen_test_mse, unseen_test_mse,
                train_mae, valid_mae, seen_test_mae, unseen_test_mae,
                train_mape, valid_mape, seen_test_mape, unseen_test_mape,
                train_r2, valid_r2, seen_test_r2, unseen_test_r2,
                train_pccs_rho, valid_pccs_rho, seen_test_pccs_rho, unseen_test_pccs_rho,
                train_spear_rho, valid_spear_rho, seen_test_spear_rho, unseen_test_spear_rho
            ]

            if valid_mse < self.min_dif:
                self.min_dif = valid_mse
                self.best_res = res_list
                self.best_epoch = epoch
                # save model
                # same_model_param_path = osp.join(current_path, save_model_data_dir, self.trainer_info + '_param.pkl')
                # torch.save(self.model.state_dict(), same_model_param_path)

            if display:
                self.print_res(res_list, epoch)

            if epoch % 50 == 0 and epoch > 0:
                print('Best res')
                self.print_res(self.best_res, self.best_epoch)


    def evaluate_model(self):
        pass

if __name__ == '__main__':
    parser = ArgumentParser(description="Train model")
    parser.add_argument('--mode', type=str, default='evaluate', help='train or evaluate')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    if args.mode == 'train':
        for seed in range(20):
            print('seed = ', seed)
            param_dict = {
                'hot_data_split': [0.9, 0.05, 0.05],
                'kmer_min_df': 0.1,
                'seed': 0,
                'batch_size': 32,
                'epoch_num': 200,
                'h_dim': 512,
                'dropout_num': 0.4,
                'lr': 1e-4,
                'amino_embedding_dim': 7,
                'adj_loss_coef': 5e-4,
                'param_l1_coef': 0,
                'param_l2_coef': 5e-4,
                'add_res': True,
                'add_bn': False,
                'max_antibody_len': 344,
                'max_virus_len': 912,

            }
            trainer = Trainer(**param_dict)
            trainer.train()
    else:
        for seed in range(20):
            print('seed = ', seed)
            param_dict = {
                'hot_data_split': [0.9, 0.05, 0.05],
                'kmer_min_df': 0.1,
                'seed': 0,
                'batch_size': 32,
                'epoch_num': 200,
                'h_dim': 512,
                'dropout_num': 0.4,
                'lr': 1e-4,
                'amino_embedding_dim': 7,
                'adj_loss_coef': 5e-4,
                'param_l1_coef': 0,
                'param_l2_coef': 5e-4,
                'add_res': True,
                'add_bn': False,
                'max_antibody_len': 344,
                'max_virus_len': 912,

            }
            trainer = Trainer(**param_dict)
            trainer.evaluate_model()