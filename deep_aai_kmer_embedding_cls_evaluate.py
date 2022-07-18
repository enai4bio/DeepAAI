#!/usr/bin/env
# coding:utf-8

from argparse import ArgumentParser
from dataset.abs_dataset_cls import AbsDataset
from metrics.evaluate import evaluate_classification_paper
from models.deep_aai_kmer_embedding_cls import DeepAAIKmerEmbeddingCls
import numpy as np
import torch.nn as nn
import random
import torch
import math
import os.path as osp
from utils.index_map import get_map_index_for_sub_arr
import pandas as pd
from dataset.amino_seq_to_ft import amino_seq_to_kmer, amino_seq_to_num
from dataset.dataset_tools import get_padding_ft_dict
_, _, amino_map_idx = get_padding_ft_dict()

current_path = osp.dirname(osp.realpath(__file__))


class DeepAAIPred(object):
    def __init__(self, test_file_path, result_file_path):
        self.param_dict = {
            'seed': 2,
            'kmer_min_df': 0.1,
            'label_type': 'label_10',
            'batch_size': 32,
            'h_dim': 512,
            'dropout_num': 0.4,
            'lr': 5e-5,
            'adj_loss_coef': 5e-4,
            'param_l2_coef': 5e-4,
            'amino_embedding_dim': 7,
            'kernel_cfg': [7, 9, 11],
            'channel_cfg': [256, 256, 256],
            'dilation_cfg': [1, 1, 1],
            'add_res': True,
            'add_bn': False,
            'max_antibody_len': 344,
            'max_virus_len': 912,
        }
        self.setup_seed(self.param_dict['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_file_path = test_file_path
        self.result_file_path = result_file_path
        self.ft_dict = {}
        self.load_data()
        self.build_model()

    def load_data(self):
        # load test data -->  kmer_ft, amino_num
        test_data_df = pd.read_csv(self.test_file_path)
        if 'label_10' in test_data_df:
            self.label_mat = test_data_df['label_10'].to_numpy()
        else:
            self.label_mat = None

        virus_seq_list = test_data_df['virus_seq'].to_list()
        virus_kmer_ft = amino_seq_to_kmer(virus_seq_list)
        self.ft_dict['test_virus_kmer'] = torch.FloatTensor(virus_kmer_ft).to(self.device)
        virus_amino_num_mat = amino_seq_to_num(virus_seq_list, protein_type='virus')
        self.ft_dict['test_virus_amino_num'] = torch.LongTensor(virus_amino_num_mat).to(self.device)

        antibody_seq_list = test_data_df['heavy_seq'].str.cat(test_data_df['light_seq']).to_list()
        antibody_kmer_ft = amino_seq_to_kmer(antibody_seq_list)
        self.ft_dict['test_antibody_kmer'] = torch.FloatTensor(antibody_kmer_ft).to(self.device)
        antibody_amino_num_mat = amino_seq_to_num(antibody_seq_list, protein_type='antibody')
        self.ft_dict['test_antibody_amino_num'] = torch.LongTensor(antibody_amino_num_mat).to(self.device)

        # load bg graph
        antibody_graph_node_kmer_ft = torch.load(
            osp.join(current_path, 'dataset', 'processed_mat', 'antibody_graph_node_kmer_ft.pkl')
        )
        self.ft_dict['antibody_graph_node_kmer_ft'] = torch.FloatTensor(antibody_graph_node_kmer_ft).to(self.device)

        virus_graph_node_kmer_ft = torch.load(
            osp.join(current_path, 'dataset', 'processed_mat', 'virus_graph_node_kmer_ft.pkl')
        )
        self.ft_dict['virus_graph_node_kmer_ft'] = torch.FloatTensor(virus_graph_node_kmer_ft).to(self.device)

        self.param_dict['kmer_dim'] = self.ft_dict['test_antibody_kmer'].shape[-1]
        self.param_dict['amino_type_num'] = max(amino_map_idx.values()) + 1
        self.param_dict['max_antibody_len'] = self.ft_dict['test_antibody_amino_num'].shape[1]
        self.param_dict['max_virus_len'] = self.ft_dict['test_antibody_amino_num'].shape[1]


    def build_model(self):
        self.model = DeepAAIKmerEmbeddingCls(**self.param_dict).to(self.device)

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def prediction(self):
        self.model.eval()
        all_pred = []
        bg_virus_graph_node_num = self.ft_dict['virus_graph_node_kmer_ft'].size()[0]
        bg_antibody_graph_node_num = self.ft_dict['antibody_graph_node_kmer_ft'].size()[0]
        test_pair_num = self.ft_dict['test_antibody_kmer'].size()[0]

        range_idx = np.arange(0, test_pair_num)
        for i in range(math.ceil(test_pair_num/self.param_dict['batch_size'])):
            right_bound = min((i + 1)*self.param_dict['batch_size'], test_pair_num + 1)
            batch_idx = range_idx[i * self.param_dict['batch_size']: right_bound]

            batch_antibody_amino_ft = self.ft_dict['test_antibody_amino_num'][batch_idx]
            batch_virus_amino_ft = self.ft_dict['test_virus_amino_num'][batch_idx]

            batch_test_antibody_kmer_ft = self.ft_dict['test_antibody_kmer'][batch_idx]
            batch_test_virus_kmer_ft = self.ft_dict['test_virus_kmer'][batch_idx]

            all_antibody_kmer_ft = torch.cat(
                [self.ft_dict['antibody_graph_node_kmer_ft'], batch_test_antibody_kmer_ft], dim=0)
            all_virus_kmer_ft = torch.cat(
                [self.ft_dict['virus_graph_node_kmer_ft'], batch_test_virus_kmer_ft], dim=0)

            batch_antibody_node_idx_in_graph = np.arange(0, batch_idx.shape[0]) + bg_antibody_graph_node_num
            batch_virus_node_idx_in_graph = np.arange(0, batch_idx.shape[0]) + bg_virus_graph_node_num

            model_input_ft_dict = {
                'antibody_graph_node_kmer_ft': all_antibody_kmer_ft,
                'virus_graph_node_kmer_ft': all_virus_kmer_ft,
                'antibody_amino_ft': batch_antibody_amino_ft,
                'virus_amino_ft': batch_virus_amino_ft,
                'antibody_idx': batch_antibody_node_idx_in_graph,
                'virus_idx': batch_virus_node_idx_in_graph
            }

            pred, _, _ = self.model(**model_input_ft_dict)
            pred = pred.view(-1)
            pred = pred.detach().to('cpu').numpy()
            all_pred = np.hstack([all_pred, pred])
        return all_pred

    def evaluate_model(self):
        # load param
        model_file_path = osp.join(current_path, 'save_model_param_pred', 'deep_aai_k+e',
                                   'deep_aai_kmer_embedding_cls_seed=2_param.pkl')
        self.model.load_state_dict(torch.load(model_file_path))
        print('load_param ', model_file_path)

        test_pred = self.prediction()
        data_df = pd.read_csv(self.test_file_path)
        data_df['pred'] = test_pred
        data_df.to_csv(self.result_file_path)

        if self.label_mat is not None:
            test_acc, test_f1, test_roc_auc, test_pr_auc = \
                evaluate_classification_paper(predict_proba=test_pred, label=self.label_mat)

            log_str = \
                'Evaluate Result:  ACC      \tF1     \tROC-AUC \tPR-AUC  \n' \
                'Unseen Test:      {:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(
                test_acc, test_f1, test_roc_auc, test_pr_auc)
            print(log_str)



if __name__ == '__main__':
    parser = ArgumentParser(description="Train model")
    parser.add_argument('--infile', type=str, default='test_data.csv', help='test csv file')
    parser.add_argument('--outfile', type=str, default='pred_result.csv', help='test csv file')
    args = parser.parse_args()
    model = DeepAAIPred(args.infile, args.outfile)
    model.evaluate_model()





