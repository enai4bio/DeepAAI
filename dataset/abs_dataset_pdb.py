#!/usr/bin/env
# coding:utf-8
"""
Created on 2020/8/21 14:13

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import os.path as osp
import numpy as np
import pandas as pd
import torch
from dataset.dataset_tools import get_padding_ft_dict, get_index_in_target_list
from dataset.k_mer_utils import KmerTranslator
from utils.sorted_relate import set_sort_by_select_type
from dataset.dataset_split import train_test_split
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

amino_one_hot_ft_pad_dict, amino_physicochemical_ft_pad_dict, amino_map_idx = get_padding_ft_dict()
current_path = osp.dirname(osp.realpath(__file__))
corpus_dir = 'corpus'
data_version = 'split_anti'
pdb_data_dir = 'pdb'
pdb_file = osp.join(current_path, corpus_dir, pdb_data_dir, 'pdb_data_add_hiv_unseen_neg_pair.csv')


class PDBDataset():
    def __init__(self,
                 max_antibody_len='auto',
                 max_virus_len='auto',
                 select_type='all',
                 generate_negative_pair=False,
                 kmer_min_df=1,
                 split_param=[0.8, 0.2]
                 ):

        self.generate_negative_pair = generate_negative_pair
        self.dataset_select_type = select_type
        # load hiv-unseen-negative-pair --> negative sample
        self.dataset_select_type.append('hiv_unseen_neg')

        self.max_antibody_len = max_antibody_len
        self.max_virus_len = max_virus_len
        self.kmer_min_df = kmer_min_df
        self.processed_kmer_name = 'abs_dataset_cls_antibody=344_virus=912_kmer_min_df={}'.format(self.kmer_min_df)

        self.protein_ft_dict = {}
        self.split_param = split_param
        self.all_label_mat = None  # 1: pair in pdb,  0: unpair in hiv

        self.raw_all_antibody_name = None
        self.raw_all_virus_name = None

        self.raw_all_antibody_set = None
        self.raw_all_virus_set = None

        self.raw_all_antibody_set_len = []
        self.raw_all_virus_set_len = []

        self.antibody_index_in_pair = None
        self.virus_index_in_pair = None

        self.train_index = None
        self.valid_index = None

        self.load_file()

        self.generate_ft_mat()

    def to_tensor(self, device='cpu'):
        # print('sara protein ft to tensor ', device)
        for ft_name in self.protein_ft_dict:
            # print('sars to device ', ft_name)
            if ft_name in ['antibody_amino_num', 'virus_amino_num']:
                self.protein_ft_dict[ft_name] = torch.LongTensor(self.protein_ft_dict[ft_name]).to(device)
            else:
                self.protein_ft_dict[ft_name] = torch.FloatTensor(self.protein_ft_dict[ft_name]).to(device)

    def load_file(self):
        # load dataset
        pdb_data = pd.read_csv(pdb_file, index_col=0)
        if self.dataset_select_type != 'all':
            pdb_data = pdb_data[pdb_data['split_type'].isin(self.dataset_select_type)].reset_index()
        # print(pdb_data)

        # get antibody set
        all_h_chain_name = pdb_data['Hchain'].to_list()
        all_l_chain_name = pdb_data['Lchain'].to_list()
        all_split_type = pdb_data['split_type'].to_list()
        all_virus_name = pdb_data['antigen_chain'].to_list()

        all_antibody_name = [all_h_chain_name[idx].split('_')[0] + '__' + all_l_chain_name[idx].split('_')[0] + '__' + all_split_type[idx] for idx in range(pdb_data.shape[0])]
        self.raw_all_antibody_name = set_sort_by_select_type(all_antibody_name, all_split_type)
        # self.raw_all_antibody_name = np.array(self.raw_all_antibody_name)
        antibody_idx = get_index_in_target_list(self.raw_all_antibody_name, all_antibody_name)
        self.raw_all_antibody_set = [
            pdb_data.loc[idx]['Hchain_seq_fv'] + pdb_data.loc[idx]['Lchain_seq_fv'] for idx in antibody_idx
        ]

        # get virus seq
        all_virus_name = [all_virus_name[idx].split('_')[0] + '__' + all_split_type[idx] for idx in range(pdb_data.shape[0])]
        self.raw_all_virus_name = set_sort_by_select_type(all_virus_name, all_split_type)
        # self.raw_all_virus_name = np.array(self.raw_all_virus_name)
        # print('len(self.raw_all_virus_name) = ', len(self.raw_all_virus_name), len(all_virus_name))
        virus_idx = get_index_in_target_list(self.raw_all_virus_name, all_virus_name)
        self.raw_all_virus_set = pdb_data.loc[virus_idx]['antigen_seq'].to_list()
        self.all_virus_type = pdb_data.loc[virus_idx]['split_type'].to_list()

        print(len(self.raw_all_virus_set), len(self.raw_all_antibody_set))
        # exit()
        if self.max_virus_len == 'auto':
            self.max_virus_len = 0
            for idx in range(len(self.raw_all_virus_set)):
                s = self.raw_all_virus_set[idx]
                self.max_virus_len = max(self.max_virus_len, len(s))
                # print('len virus ', len(s),  self.all_virus_type[idx], self.raw_all_virus_name[idx])
        if self.max_antibody_len == 'auto':
            self.max_antibody_len = 0
            for s in self.raw_all_antibody_set:
                self.max_antibody_len = max(self.max_antibody_len, len(s))

        for s in self.raw_all_antibody_set:
            self.raw_all_antibody_set_len.append(min(len(s), self.max_antibody_len))
        for s in self.raw_all_virus_set:
            self.raw_all_virus_set_len.append(min(len(s), self.max_virus_len))

        # built pair  positive: pair in file,   negative: not in file
        # print('all_antibody_name = ', all_antibody_name)
        # print('self.raw_all_antibody_name = ', self.raw_all_antibody_name)
        self.antibody_index_in_pair = get_index_in_target_list(all_antibody_name, self.raw_all_antibody_name)
        self.virus_index_in_pair = get_index_in_target_list(all_virus_name, self.raw_all_virus_name)

        all_split_type = np.array(all_split_type)
        self.all_label_mat = (all_split_type == 'hiv_unseen_neg').astype(np.long)

        # same num   pos_sample neg_sample
        pos_pair_idx = np.where(self.all_label_mat == 0)[0]
        neg_pair_idx = np.where(self.all_label_mat == 1)[0]

        if neg_pair_idx.shape[0] > pos_pair_idx.shape[0]:
            neg_pair_idx = np.random.choice(neg_pair_idx, size=pos_pair_idx.shape[0])

        print('pos_pair_idx = ', pos_pair_idx.shape, pos_pair_idx)
        print('neg_pair_idx = ', neg_pair_idx.shape, neg_pair_idx)

        self.split_param.append(0)

        pos_train_index, pos_valid_index, _ = train_test_split(
            record_num=pos_pair_idx.shape[0], split_param=self.split_param, extra_to='train',
            shuffle=True)
        neg_train_index, neg_valid_index, _ = train_test_split(
            record_num=neg_pair_idx.shape[0], split_param=self.split_param, extra_to='train',
            shuffle=True)

        self.train_index = np.concatenate([pos_pair_idx[pos_train_index], neg_pair_idx[neg_train_index]], axis=0)
        self.valid_index = np.concatenate([pos_pair_idx[pos_valid_index], neg_pair_idx[neg_valid_index]], axis=0)

        # print('self.train_index = ', self.train_index.shape, self.train_index)
        # print('self.valid_index = ', self.valid_index.shape, self.valid_index)
    #
    def generate_ft_mat(self):
        # one-hot
        self.protein_ft_dict['antibody_one_hot'] = self.protein_seq_list_to_ft_mat(
            self.raw_all_antibody_set, self.max_antibody_len, ft_type='amino_one_hot')
        self.protein_ft_dict['virus_one_hot'] = self.protein_seq_list_to_ft_mat(
            self.raw_all_virus_set, self.max_virus_len, ft_type='amino_one_hot')

        # amino_num
        self.protein_ft_dict['antibody_amino_num'] = self.protein_seq_list_to_ft_mat(
            self.raw_all_antibody_set, self.max_antibody_len, ft_type='amino_num')
        self.protein_ft_dict['virus_amino_num'] = self.protein_seq_list_to_ft_mat(
            self.raw_all_virus_set, self.max_virus_len, ft_type='amino_num')

        # k-mer-whole
        kmer_translator = KmerTranslator(name=self.processed_kmer_name)
        kmer_translator.load()
        self.protein_ft_dict['antibody_kmer_whole'] = kmer_translator.transform(self.raw_all_antibody_set)
        self.protein_ft_dict['virus_kmer_whole'] = kmer_translator.transform(self.raw_all_virus_set)

    def protein_seq_list_to_ft_mat(self, protein_seq_list, max_sql_len, ft_type='amino_one_hot'):
        '''
        Construct amino acid features according to select_protein_ft
        :param protein_seq_list:  list --- amino acid sequences with different lengths
        :return:  np_array    n * max_len * ft_size
        '''
        ft_mat = []
        for protein_seq in protein_seq_list:
            # append amino ft
            protein_ft = []
            for idx in range(max_sql_len):
                if idx < len(protein_seq):
                    amino_name = protein_seq[idx]
                else:
                    amino_name = 'pad'

                if 'amino_one_hot' == ft_type:
                    amino_ft = amino_one_hot_ft_pad_dict[amino_name]
                elif 'phych' == ft_type:
                    amino_ft = amino_physicochemical_ft_pad_dict[amino_name]
                elif 'amino_num' == ft_type:
                    amino_ft = amino_map_idx[amino_name]
                else:
                    exit('error ft_type')
                protein_ft.append(amino_ft)

            protein_ft = np.array(protein_ft)
            ft_mat.append(protein_ft)

        ft_mat = np.array(ft_mat).astype(np.float32)
        return ft_mat


if __name__ == '__main__':
    # virus_species_name_list = [
    #     'ebola', 'coxsackievirus', 'dengue', 'eastern equine encephalitis', 'hepa', 'astrovirus', 'cytomegalovirus',
    #     'herpesvirus', 'immunodeficiency', 'respiratory syncytial', 'influenza a', 'junin mammarenavirus',
    #     'middle east respiratory syndrome',
    #     'nipah virus', 'sars coronavirus', 'vaccinia virus', 'west nile virus', 'zika virus', 'sars_jinru'
    # ]
    dataset = PDBDataset(select_type=['dengue'], generate_negative_pair=False, kmer_min_df=0.1, split_param=[0.5, 0.5])

