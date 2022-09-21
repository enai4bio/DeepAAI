#!/usr/bin/env
# coding:utf-8

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TorchAbsDataset(Dataset):
    def __init__(self, abs_dataset, split_type, ft_type):
        assert split_type in ['train', 'valid', 'seen_test', 'unseen_test']
        assert ft_type in ['kmer_whole', 'amino_num', 'pssm', 'one_hot']

        self.abs_dataset = abs_dataset
        self.split_type = split_type
        self.ft_type = ft_type

        if self.split_type == 'train':
            self.select_pair_idx = self.abs_dataset.train_index
        elif self.split_type == 'valid':
            self.select_pair_idx = self.abs_dataset.valid_seen_index
        elif self.split_type == 'seen_test':
            self.select_pair_idx = self.abs_dataset.test_seen_index
        else:
            self.select_pair_idx = self.abs_dataset.test_unseen_index

    def __len__(self):
        return self.select_pair_idx.shape[0]

    def __getitem__(self, idx):
        pair_idx = self.select_pair_idx[idx]

        antibody_idx = self.abs_dataset.antibody_index_in_pair[pair_idx]
        virus_idx = self.abs_dataset.virus_index_in_pair[pair_idx]
        antibody_ft = self.abs_dataset.protein_ft_dict['antibody_'+self.ft_type][antibody_idx]
        virus_ft = self.abs_dataset.protein_ft_dict['virus_'+self.ft_type][virus_idx]
        pair_label = self.abs_dataset.all_label_mat[pair_idx]

        return antibody_ft, virus_ft, pair_label


class TorchAbsDatasetParapred(Dataset):
    def __init__(self, abs_dataset, split_type, ft_type):
        assert split_type in ['train', 'valid', 'seen_test', 'unseen_test']
        assert ft_type in ['kmer_whole', 'amino_num', 'pssm', 'one_hot']

        self.abs_dataset = abs_dataset
        self.split_type = split_type
        self.ft_type = ft_type

        if self.split_type == 'train':
            self.select_pair_idx = self.abs_dataset.train_index
        elif self.split_type == 'valid':
            self.select_pair_idx = self.abs_dataset.valid_seen_index
        elif self.split_type == 'seen_test':
            self.select_pair_idx = self.abs_dataset.test_seen_index
        else:
            self.select_pair_idx = self.abs_dataset.test_unseen_index

    def __len__(self):
        return self.select_pair_idx.shape[0]

    def __getitem__(self, idx):
        pair_idx = self.select_pair_idx[idx]

        antibody_idx = self.abs_dataset.antibody_index_in_pair[pair_idx]
        virus_idx = self.abs_dataset.virus_index_in_pair[pair_idx]
        antibody_ft = self.abs_dataset.protein_ft_dict['antibody_'+self.ft_type][antibody_idx]
        virus_ft = self.abs_dataset.protein_ft_dict['virus_'+self.ft_type][virus_idx]
        pair_label = self.abs_dataset.all_label_mat[pair_idx]

        antibody_len = self.abs_dataset.raw_all_antibody_set_len[antibody_idx]
        virus_len = self.abs_dataset.raw_all_virus_set_len[virus_idx]

        return antibody_ft, virus_ft, pair_label, antibody_len, virus_len


class TorchAbsDatasetMaya(Dataset):
    def __init__(self, abs_dataset, split_type, ft_type, ):
        assert split_type in ['train', 'valid', 'seen_test', 'unseen_test']
        assert ft_type in ['kmer_whole', 'amino_num', 'pssm', 'one_hot']

        self.abs_dataset = abs_dataset
        self.split_type = split_type
        self.ft_type = ft_type

        if self.split_type == 'train':
            self.select_pair_idx = self.abs_dataset.train_index
        elif self.split_type == 'valid':
            self.select_pair_idx = self.abs_dataset.valid_seen_index
        elif self.split_type == 'seen_test':
            self.select_pair_idx = self.abs_dataset.test_seen_index
        else:
            self.select_pair_idx = self.abs_dataset.test_unseen_index


    def __len__(self):
        return self.select_pair_idx.shape[0]

    def __getitem__(self, idx):
        pair_idx = self.select_pair_idx[idx]
        antibody_idx = self.abs_dataset.antibody_index_in_pair[pair_idx]
        virus_idx = self.abs_dataset.virus_index_in_pair[pair_idx]
        antibody_ft = self.abs_dataset.protein_ft_dict['antibody_'+self.ft_type][antibody_idx]
        virus_ft = self.abs_dataset.protein_ft_dict['virus_'+self.ft_type][virus_idx]
        pair_label = self.abs_dataset.all_label_mat[pair_idx]
        return antibody_ft, virus_ft, pair_label, antibody_idx, virus_idx


class TorchPdbDatasetParapred(Dataset):
    def __init__(self, abs_dataset, split_type):
        assert split_type in ['train', 'valid']
        self.abs_dataset = abs_dataset
        self.split_type = split_type

        if self.split_type == 'train':
            self.select_pair_idx = self.abs_dataset.train_index
        else:
            self.select_pair_idx = self.abs_dataset.valid_index

    def __len__(self):
        return self.select_pair_idx.shape[0]

    def __getitem__(self, idx):
        pair_idx = self.select_pair_idx[idx]

        antibody_idx = self.abs_dataset.antibody_index_in_pair[pair_idx]
        virus_idx = self.abs_dataset.virus_index_in_pair[pair_idx]
        antibody_ft = self.abs_dataset.protein_ft_dict['antibody_one_hot'][antibody_idx]
        virus_ft = self.abs_dataset.protein_ft_dict['virus_one_hot'][virus_idx]
        pair_label = self.abs_dataset.all_label_mat[pair_idx]

        antibody_len = self.abs_dataset.raw_all_antibody_set_len[antibody_idx]
        virus_len = self.abs_dataset.raw_all_virus_set_len[virus_idx]

        return antibody_ft, virus_ft, pair_label, antibody_len, virus_len