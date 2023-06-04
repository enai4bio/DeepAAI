 #!/usr/bin/env
# coding:utf-8

import os.path as osp
import numpy as np
import pandas as pd
import torch
import json
from dataset.dataset_tools import get_padding_ft_dict, get_index_in_target_list
from dataset.dataset_split import train_test_split
from dataset.k_mer_utils import KmerTranslator

try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

current_path = osp.dirname(osp.realpath(__file__))
corpus_dir = 'corpus/reg'
processed_dir = 'corpus/processed_mat'

amino_one_hot_ft_pad_dict, amino_physicochemical_ft_pad_dict, amino_map_idx = get_padding_ft_dict()


class AbsDataset():
    def __init__(self, max_antibody_len=344, max_virus_len=912,
                 train_split_param=[0.9, 0.05, 0.05],
                 reprocess=False,
                 kmer_min_df=0.1,
                 ):
        self.dataset_name = __file__.split('/')[-1].replace('.py', '')
        self.dataset_param_str = '{}_antibody={}_virus={}_kmer_min_df={}'.format(
            self.dataset_name, max_antibody_len, max_virus_len, kmer_min_df)
        self.protein_ft_save_path = osp.join(current_path, processed_dir, self.dataset_param_str+'__protein_ft_dict.pkl')
        self.reporcess = reprocess
        self.train_split_param = train_split_param
#         self.label_type = label_type
        self.kmer_min_df = kmer_min_df
        self.max_antibody_len = max_antibody_len
        self.max_virus_len = max_virus_len

        self.dataset_info = {}
        self.protein_ft_dict = {}
        # key: (antibody_heavy or antibody_light or virus)_(ft: one_hot or pssm or k_mer)

        self.all_label_mat = np.load(osp.join(current_path, corpus_dir, 'all_label_mat.npy'))

        self.antibody_index_in_pair = np.load(osp.join(current_path, corpus_dir, 'antibody_index_in_pair.npy'))
        self.virus_index_in_pair = np.load(osp.join(current_path, corpus_dir, 'virus_index_in_pair.npy'))

        self.known_antibody_idx = np.load(osp.join(current_path, corpus_dir, 'known_antibody_idx.npy'))
        self.known_virus_idx = np.load(osp.join(current_path, corpus_dir, 'known_virus_idx.npy'))
        self.unknown_antibody_idx = np.load(osp.join(current_path, corpus_dir, 'unknown_antibody_idx.npy'))

        self.train_index = np.load(osp.join(current_path, corpus_dir, 'train_index.npy'))
        self.valid_seen_index = None
        self.test_seen_index = None
        self.test_unseen_index = np.load(osp.join(current_path, corpus_dir, 'test_unseen_index.npy'))
        
        self.raw_all_antibody_set_len = np.load(osp.join(current_path, corpus_dir, 'raw_all_virus_set_len.npy'))
        self.raw_all_virus_set_len = np.load(osp.join(current_path, corpus_dir, 'raw_all_virus_set_len.npy'))


        # load existing processed file
        print('loading ', self.protein_ft_save_path)
        with open(self.protein_ft_save_path, 'rb') as f:
            if sys.version_info > (3, 0):
                self.protein_ft_dict = pickle.load(f)
            else:
                self.protein_ft_dict = pickle.load(f)

        self.generate_dataset_info()

        # data split
        # seen ->  train\valid\seen_test
        # unseen -> unseen_test
        raw_train_size = self.train_index.shape[0]
        raw_train_index = self.train_index
        new_train_index, seen_valid_index, seen_test_index = train_test_split(
            record_num=raw_train_size, split_param=self.train_split_param, extra_to='train', shuffle=True)
        self.train_index = raw_train_index[new_train_index]
        self.valid_seen_index = raw_train_index[seen_valid_index]
        self.test_seen_index = raw_train_index[seen_test_index]

    def to_tensor(self, device='cpu'):
        print('hiv protein ft to tensor ', device)
        for ft_name in self.protein_ft_dict:
            if ft_name in ['antibody_amino_num', 'virus_amino_num']:
                self.protein_ft_dict[ft_name] = torch.LongTensor(self.protein_ft_dict[ft_name]).to(device)
            else:
                self.protein_ft_dict[ft_name] = torch.FloatTensor(self.protein_ft_dict[ft_name]).to(device)


    def generate_dataset_info(self):
        self.dataset_info['kmer_dim'] = self.protein_ft_dict['virus_kmer_whole'].shape[-1]
        self.dataset_info['pssm_antibody_dim'] = self.protein_ft_dict['antibody_pssm'].shape[-1]
        self.dataset_info['pssm_virus_dim'] = self.protein_ft_dict['virus_pssm'].shape[-1]
        self.dataset_info['amino_type_num'] = max(amino_map_idx.values()) + 1
        self.dataset_info['max_antibody_len'] = self.protein_ft_dict['antibody_one_hot'].shape[1]
        self.dataset_info['max_virus_len'] = self.protein_ft_dict['virus_one_hot'].shape[1]



if __name__ == '__main__':
    print(0)
