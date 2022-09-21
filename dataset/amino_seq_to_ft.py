#!/usr/bin/env
# coding:utf-8

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


def amino_seq_to_kmer(protein_seq_list, min_df=0.1):
    processed_kmer_obj_name = 'abs_dataset_cls_antibody=344_virus=912_kmer_min_df={}'.format(min_df)
    kmer_translator = KmerTranslator(name=processed_kmer_obj_name)
    kmer_translator.load()
    kmer_ft_mat = kmer_translator.transform(protein_seq_list)
    return kmer_ft_mat


def amino_seq_to_num(protein_seq_list, protein_type, antibody_max_len=344, virus_max_len=912):
    assert protein_type in ['virus', 'antibody']
    if protein_type == 'virus':
        max_sql_len = antibody_max_len
    else:
        max_sql_len = virus_max_len

    ft_mat = []
    for protein_seq in protein_seq_list:
        # append amino ft
        amino_num_list = []
        for idx in range(max_sql_len):
            if idx < len(protein_seq):
                amino_name = protein_seq[idx]
            else:
                amino_name = 'pad'
            amino_idx = amino_map_idx[amino_name]
            amino_num_list.append(amino_idx)
        ft_mat.append(amino_num_list)
    ft_mat = np.array(ft_mat).astype(np.long)
    return ft_mat


def amino_seq_to_one_hot(protein_seq_list, protein_type, antibody_max_len=344, virus_max_len=912):
    assert protein_type in ['virus', 'antibody']
    if protein_type == 'virus':
        max_sql_len = antibody_max_len
    else:
        max_sql_len = virus_max_len

    ft_mat = []
    for protein_seq in protein_seq_list:
        # append amino ft
        amino_ft_list = []
        for idx in range(max_sql_len):
            if idx < len(protein_seq):
                amino_name = protein_seq[idx]
            else:
                amino_name = 'pad'
            amino_ft = amino_physicochemical_ft_pad_dict[amino_name]
            amino_ft_list.append(amino_ft)
        ft_mat.append(amino_ft_list)
    ft_mat = np.array(ft_mat).astype(np.float32)
    return ft_mat

