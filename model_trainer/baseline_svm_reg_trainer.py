#!/usr/bin/env
# coding:utf-8
import sys
import os
sys.path.append(os.getcwd())

from sklearn.svm import SVC, SVR
from dataset.abs_dataset_reg import AbsDataset
from metrics.evaluate import evaluate_regression_v2
import numpy as np
import random
import torch
import math
import os.path as osp
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

current_path = osp.dirname(osp.realpath(__file__))


class MLTrainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict
        self.setup_seed(self.param_dict['seed'])
        self.file_name = __file__.split('/')[-1].replace('.py', '')
        self.trainer_info = '{}_seed={}_label_type={}'.format(self.file_name, self.param_dict['seed'],
                                                         self.param_dict['label_type'])
        self.save_model_path = osp.join(current_path, '..', 'save_model_param_pred', self.trainer_info+'_model.pkl')

        self.dataset = AbsDataset(max_antibody_len=self.param_dict['max_antibody_len'],
                                  max_virus_len=self.param_dict['max_virus_len'],
                                  train_split_param=self.param_dict['hot_data_split'],
                                  reprocess=True)

        self.model = SVR(C=self.param_dict['svm_c'], gamma=self.param_dict['svm_gamma'])
        self.pair_num = self.dataset.all_label_mat.shape[0]

        # Multiple features  cat reshape to  (batch, -1)
        # construct ft mat
        self.all_ft_mat = []
        for ft_type in self.param_dict['select_ft']:
            if ft_type == 'one_hot':
                antibody_ft = self.dataset.protein_ft_dict['antibody_one_hot'][self.dataset.antibody_index_in_pair].reshape(self.pair_num, -1)
                virus_ft = self.dataset.protein_ft_dict['virus_one_hot'][self.dataset.virus_index_in_pair].reshape(self.pair_num, -1)
            if ft_type == 'pssm':
                antibody_ft = self.dataset.protein_ft_dict['antibody_pssm'][self.dataset.antibody_index_in_pair].reshape(self.pair_num, -1)
                virus_ft = self.dataset.protein_ft_dict['virus_pssm'][self.dataset.virus_index_in_pair].reshape(self.pair_num, -1)
            if ft_type == 'kmer':
                antibody_ft = self.dataset.protein_ft_dict['antibody_kmer_whole'][self.dataset.antibody_index_in_pair].reshape(self.pair_num, -1)
                virus_ft = self.dataset.protein_ft_dict['virus_kmer_whole'][self.dataset.virus_index_in_pair].reshape(self.pair_num, -1)

            self.all_ft_mat.append(np.concatenate((antibody_ft, virus_ft), axis=1))
            print(self.all_ft_mat[-1].shape)
        self.all_ft_mat = np.concatenate(self.all_ft_mat, axis=1)

    def fit(self):
        self.model.fit(self.all_ft_mat[self.dataset.train_index], self.dataset.all_label_mat[self.dataset.train_index])

    def pred(self):
        train_pred = self.model.predict(self.all_ft_mat[self.dataset.train_index])
        valid_seen_pred = self.model.predict(self.all_ft_mat[self.dataset.valid_seen_index])
        test_seen_pred = self.model.predict(self.all_ft_mat[self.dataset.test_seen_index])
        test_unseen_pred = self.model.predict(self.all_ft_mat[self.dataset.test_unseen_index])

        print('mse, mae, mape, r2, pccs_rho, pccs_pval, spear_rho, spear_pval')
        print('train ', evaluate_regression_v2(
            pred=train_pred, label=self.dataset.all_label_mat[self.dataset.train_index]))

        print('seen valid ', evaluate_regression_v2(
            pred=valid_seen_pred, label=self.dataset.all_label_mat[self.dataset.valid_seen_index]))

        print('seen test ', evaluate_regression_v2(
            pred=test_seen_pred, label=self.dataset.all_label_mat[self.dataset.test_seen_index]))

        print('unseen test ', evaluate_regression_v2(
            pred=test_unseen_pred, label=self.dataset.all_label_mat[self.dataset.test_unseen_index]))

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def save_model(self):
        with open(self.save_model_path, 'wb') as f:
            if sys.version_info > (3, 0):
                pickle.dump(self.model, f)
            else:
                pickle.dump(self.model, f)

    def load_model(self):
        print('loading ', self.save_model_path)
        with open(self.save_model_path, 'rb') as f:
            if sys.version_info > (3, 0):
                self.model = pickle.load(f)
            else:
                self.model = pickle.load(f)

if __name__ == '__main__':
    for seed in range(20):
        print('seed = ', seed)
        param_dict = {
            'hot_data_split': [0.9, 0.05, 0.05],
            'seed': seed,
            'max_antibody_len': 344,
            'max_virus_len': 912,
            'select_ft': ['pssm'],
            'svm_c': 32,
            'svm_gamma': 0.03125
        }
        trainer = MLTrainer(**param_dict)
        trainer.fit()
        # trainer.load_model()
        trainer.pred()
        # trainer.save_model()
