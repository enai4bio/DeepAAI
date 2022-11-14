#!/usr/bin/env
# coding:utf-8
import sys
import os
sys.path.append(os.getcwd())

from argparse import ArgumentParser
import torch.nn as nn
from dataset.abs_dataset_cls import AbsDataset
from metrics.evaluate import evaluate_classification
from models.deep_aai_kmer_embedding_cls import DeepAAIKmerEmbeddingCls
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
                                  label_type=self.param_dict['label_type'],
                                  kmer_min_df=self.param_dict['kmer_min_df'],
                                  reprocess=False)
        self.param_dict.update(self.dataset.dataset_info)
        self.dataset.to_tensor(self.device)
        self.file_name = __file__.split('/')[-1].replace('.py', '')
        self.trainer_info = '{}_seed={}_batch={}'.format(self.file_name, self.param_dict['seed'], self.param_dict['batch_size'])
        self.loss_op = nn.BCELoss()
        self.build_model()

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
        # antibody_graph_node_pssm_ft = self.dataset.protein_ft_dict['antibody_pssm'][antibody_graph_node_idx]
        # virus_graph_node_pssm_ft = self.dataset.protein_ft_dict['virus_pssm'][virus_graph_node_idx]

        # torch.save(antibody_graph_node_kmer_ft.to('cpu'), 'antibody_graph_node_kmer_ft.pkl')
        # torch.save(virus_graph_node_kmer_ft.to('cpu'), 'virus_graph_node_kmer_ft.pkl')
        # exit()

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
                'antibody_graph_node_kmer_ft': antibody_graph_node_kmer_ft,
                'virus_graph_node_kmer_ft': virus_graph_node_kmer_ft,
                # 'antibody_graph_node_pssm_ft': antibody_graph_node_pssm_ft,
                # 'virus_graph_node_pssm_ft': virus_graph_node_pssm_ft,
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
                # param_l1_loss = self.param_dict['param_l1_coef'] * param_l1_loss
                adj_l1_loss = self.param_dict['adj_loss_coef'] * torch.norm(virus_adj) + \
                              self.param_dict['adj_loss_coef'] * torch.norm(antibody_adj)
                loss = c_loss + adj_l1_loss + param_l2_loss
                # print('c_loss = ', c_loss.detach().to('cpu').numpy(),
                #       'adj_l1_loss = ', adj_l1_loss.detach().to('cpu').numpy(),
                #       'param_l2_loss = ', param_l2_loss.detach().to('cpu').numpy())

                all_loss.append(loss.detach().to('cpu').item())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            pred = pred.detach().to('cpu').numpy()
            all_pred = np.hstack([all_pred, pred])
            all_label = np.hstack([all_label, batch_label]).astype(np.long)
        # print(all_loss)

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
                # save model
                # save_complete_model_path = osp.join(current_path, save_model_data_dir, self.trainer_info + '_complete.pkl')
                # torch.save(self.model, save_complete_model_path)
                # same_model_param_path = osp.join(current_path, save_model_data_dir, self.trainer_info + '_param.pkl')
                # torch.save(self.model.state_dict(), same_model_param_path)

            if display:
                self.print_res(res_list, epoch)

            if epoch % 50 == 0 and epoch > 0:
                print('Best res')
                self.print_res(self.best_res, self.best_epoch)

    def evaluate_model(self):
        # load param
        model_file_path = osp.join(current_path, '..', 'save_model_param_pred',
                                   'deep_aai_k+e', 'deep_aai_kmer_embedding_cls_seed={}_param.pkl'.format(self.param_dict['seed']))
        self.model.load_state_dict(torch.load(model_file_path))
        print('load_param ', model_file_path)

        seen_test_pred, seen_test_label, _ = self.iteration(
            0, self.dataset.test_seen_index,
            antibody_graph_node_idx=self.dataset.known_antibody_idx,
            virus_graph_node_idx=self.dataset.known_virus_idx,
            is_training=False, shuffle=False)
        seen_test_acc, seen_test_p, seen_test_r, seen_test_f1, seen_test_auc, seen_test_mcc = \
            evaluate_classification(predict_proba=seen_test_pred, label=seen_test_label)

        unseen_test_pred, unseen_test_label, _ = self.iteration(
            0, self.dataset.test_unseen_index,
            antibody_graph_node_idx=np.hstack((self.dataset.known_antibody_idx, self.dataset.unknown_antibody_idx)),
            virus_graph_node_idx=self.dataset.known_virus_idx,
            is_training=False, shuffle=False)
        unseen_test_acc, unseen_test_p, unseen_test_r, unseen_test_f1, unseen_test_auc, unseen_test_mcc = \
            evaluate_classification(predict_proba=unseen_test_pred, label=unseen_test_label)

        log_str = \
            'Evaluate Result:  ACC      \tP      \tR      \tF1     \tAUC    \tMCC  \n' \
            'Seen Test:        {:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}  \n'  \
            'Unseen Test:      {:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(
            seen_test_acc, seen_test_p, seen_test_r, seen_test_f1, seen_test_auc, seen_test_mcc,
            unseen_test_acc, unseen_test_p, unseen_test_r, unseen_test_f1, unseen_test_auc, unseen_test_mcc
            )
        print(log_str)

if __name__ == '__main__':
    param_dict = {
        'hot_data_split': [0.9, 0.05, 0.05],
        'seed': 0,
        'kmer_min_df': 0.1,
        'label_type': 'label_10',
        'batch_size': 32,
        'epoch_num': 200,
        'h_dim': 512,
        'dropout_num': 0.4,
        'lr': 5e-5,
        'amino_embedding_dim': 7,
        'adj_loss_coef': 5e-4,
        'param_l2_coef': 5e-4,
        'add_res': True,
        'add_bn': False,
        'max_antibody_len': 344,
        'max_virus_len': 912,
    }

    parser = ArgumentParser(description="Train model")
    parser.add_argument('--mode', type=str, default='evaluate', help='train or evaluate')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    if args.mode == 'train':
        for seed in range(20):
            print('seed = ', seed)
            param_dict = {
                'hot_data_split': [0.9, 0.05, 0.05],
                'seed': 0,
                'kmer_min_df': 0.1,
                'label_type': 'label_10',
                'batch_size': 32,
                'epoch_num': 200,
                'h_dim': 512,
                'dropout_num': 0.4,
                'lr': 5e-5,
                'amino_embedding_dim': 7,
                'adj_loss_coef': 5e-4,
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
            param_dict = {
                'hot_data_split': [0.9, 0.05, 0.05],
                'seed': 0,
                'kmer_min_df': 0.1,
                'label_type': 'label_10',
                'batch_size': 32,
                'epoch_num': 200,
                'h_dim': 512,
                'dropout_num': 0.4,
                'lr': 5e-5,
                'amino_embedding_dim': 7,
                'adj_loss_coef': 5e-4,
                'param_l2_coef': 5e-4,
                'add_res': True,
                'add_bn': False,
                'max_antibody_len': 344,
                'max_virus_len': 912,
            }
            trainer = Trainer(**param_dict)
            trainer.evaluate_model()