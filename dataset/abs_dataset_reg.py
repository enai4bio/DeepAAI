 #!/usr/bin/env
# coding:utf-8

'''    
    medp_pssm:
        antibody (heavy light) 420 * 2
        virus 420
    
    ic50 as label
    
    
'''

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
corpus_dir = 'corpus'
data_version = 'regression_data'
processed_dir = 'processed_mat'
pssm_dir = 'hiv_pssm'

antibody_pssm_file = osp.join(current_path, corpus_dir, pssm_dir, "anti_medp_pssm_840.json")
virus_pssm_file = osp.join(current_path, corpus_dir, pssm_dir, "virus_medp_pssm_420.json")
train_file = osp.join(current_path, corpus_dir, data_version, 'data_ic50_train_v3.csv')
test_file = osp.join(current_path, corpus_dir, data_version, 'data_ic50_test_v3.csv')

amino_one_hot_ft_pad_dict, amino_physicochemical_ft_pad_dict, amino_map_idx = get_padding_ft_dict()


class AbsDataset():
    def __init__(self,
                 max_antibody_len=344,
                 max_virus_len=912,
                 kmer_min_df=0.1,
                 train_split_param=[0.9, 0.05, 0.05],
                 reprocess=False,
                 label_norm=True):

        self.dataset_name = __file__.split('/')[-1].replace('.py', '')
        self.dataset_param_str = '{}_antibody={}_virus={}_kmer_min_df={}'.format(
            self.dataset_name, max_antibody_len, max_virus_len, kmer_min_df)
        self.protein_ft_save_path = osp.join(current_path, 'processed_mat', self.dataset_param_str+'__protein_ft_dict.pkl')
        self.dataset_info = {}
        self.kmer_min_df = kmer_min_df
        self.reporcess = reprocess
        self.label_norm = label_norm
        self.train_split_param = train_split_param

        self.max_antibody_len = max_antibody_len
        self.max_virus_len = max_virus_len

        self.protein_ft_dict = {}
        # key: (antibody_heavy or antibody_light or virus)_(ft_type: one_hot pssm phych k_mer)

        self.all_label_mat = None  # ic50

        self.raw_all_antibody_name = None
        self.raw_all_virus_name = None

        self.raw_all_antibody_list = None
        self.raw_all_virus_list = None
        self.raw_all_antibody_set = None
        self.raw_all_virus_set = None

        self.raw_all_antibody_set_len = None
        self.raw_all_virus_set_len = None

        self.antibody_index_in_pair = None
        self.virus_index_in_pair = None

        self.train_index = None
        self.valid_seen_index = None
        self.test_seen_index = None
        self.test_unseen_index = None

        self.load_file()

        # try to load existing processed file
        if osp.exists(self.protein_ft_save_path) and self.reporcess is False:
            print('loading ', self.protein_ft_save_path)
            with open(self.protein_ft_save_path, 'rb') as f:
                if sys.version_info > (3, 0):
                    self.protein_ft_dict = pickle.load(f)
                else:
                    self.protein_ft_dict = pickle.load(f)
        else:
            self.generate_ft_mat()

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

        if self.label_norm is True:
            self.all_label_mat = np.log(self.all_label_mat)
        self.generate_dataset_info()

    def to_tensor(self, device='cpu'):
        # print('hiv protein ft to tensor ', device)
        for ft_name in self.protein_ft_dict:
            if ft_name in ['antibody_amino_num', 'virus_amino_num']:
                self.protein_ft_dict[ft_name] = torch.LongTensor(self.protein_ft_dict[ft_name]).to(device)
            else:
                self.protein_ft_dict[ft_name] = torch.FloatTensor(self.protein_ft_dict[ft_name]).to(device)

    def load_file(self):
        # load train(seen) test(unseen) file
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        # print(train_df)
        # antibody_name,virus_name,two_plus,virus_sequence,label,IC50

        self.train_index = np.arange(0, train_df.shape[0])
        self.test_unseen_index = np.arange(0, test_df.shape[0]) + train_df.shape[0]

        all_data_df = pd.concat([train_df, test_df], axis=0)
        self.all_label_mat = all_data_df['IC50'].to_numpy().astype(np.float32)
        self.raw_all_antibody_list = all_data_df['two_plus'].to_list()
        self.raw_all_virus_list = all_data_df['virus_sequence'].to_list()
        # print(all_data_df)

        protein_seq_map_to_name = {}
        for idx, row in all_data_df.iterrows():
            protein_seq_map_to_name[row['two_plus']] = row['antibody_name']
            protein_seq_map_to_name[row['virus_sequence']] = row['virus_name']

        self.raw_all_antibody_set = list(sorted(set(self.raw_all_antibody_list)))
        self.raw_all_virus_set = list(sorted(set(self.raw_all_virus_list)))

        self.raw_all_antibody_name = list(map(lambda s: protein_seq_map_to_name[s], self.raw_all_antibody_set))
        self.raw_all_virus_name = list(map(lambda s: protein_seq_map_to_name[s], self.raw_all_virus_set))

        # length of each amino seq
        self.raw_all_antibody_set_len = np.array(
            list(map(lambda x: len(x), self.raw_all_antibody_set)))
        self.raw_all_virus_set_len = np.array(
            list(map(lambda x: len(x), self.raw_all_virus_set)))

        self.antibody_index_in_pair = get_index_in_target_list(self.raw_all_antibody_list, self.raw_all_antibody_set)
        self.virus_index_in_pair = get_index_in_target_list(self.raw_all_virus_list, self.raw_all_virus_set)

        # known antibody idx  ---  built dynamic graph
        self.known_antibody_idx = np.unique(self.antibody_index_in_pair[self.train_index])
        self.unknown_antibody_idx = np.unique(self.antibody_index_in_pair[self.test_unseen_index])
        self.known_virus_idx = np.unique(self.virus_index_in_pair[self.train_index])


    def generate_ft_mat(self):
        # one-hot
        self.protein_ft_dict['antibody_one_hot'] = self.protein_seq_list_to_ft_mat(
            self.raw_all_antibody_set, self.max_antibody_len, ft_type='amino_one_hot')
        self.protein_ft_dict['virus_one_hot'] = self.protein_seq_list_to_ft_mat(
            self.raw_all_virus_set, self.max_virus_len, ft_type='amino_one_hot')

        # pssm
        self.protein_ft_dict['antibody_pssm'], self.protein_ft_dict['virus_pssm'] = self.load_pssm_ft_mat()

        # amino_num
        self.protein_ft_dict['antibody_amino_num'] = self.protein_seq_list_to_ft_mat(
            self.raw_all_antibody_set, self.max_antibody_len, ft_type='amino_num')
        self.protein_ft_dict['virus_amino_num'] = self.protein_seq_list_to_ft_mat(
            self.raw_all_virus_set, self.max_virus_len, ft_type='amino_num')

        # k-mer-whole
        kmer_translator = KmerTranslator(trans_type='std', min_df=self.kmer_min_df, name=self.dataset_param_str)
        protein_ft = kmer_translator.fit_transform(self.raw_all_antibody_set + self.raw_all_virus_set)
        # print('kmer protein_ft = ', protein_ft.shape, protein_ft)
        kmer_translator.save()
        self.protein_ft_dict['antibody_kmer_whole'] = protein_ft[0: len(self.raw_all_antibody_set)]
        self.protein_ft_dict['virus_kmer_whole'] = protein_ft[len(self.raw_all_antibody_set):]

        # SAVE
        print('save protein ft dict', self.protein_ft_save_path)
        with open(self.protein_ft_save_path, 'wb') as f:
            if sys.version_info > (3, 0):
                pickle.dump(self.protein_ft_dict, f)
            else:
                pickle.dump(self.protein_ft_dict, f)

    def protein_seq_list_to_ft_mat(self, protein_seq_list, max_sql_len, ft_type='amino_one_hot'):
        '''
        Construct amino acid features according to select_protein_ft
        :param protein_seq_list:  list --- amino acid sequences with different lengths
        :return:  np_array    n * max_len * ft_size
        '''
        ft_mat = []
        for protein_seq in protein_seq_list:
            # 原子特征拼接
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
        # print(ft_type, ' ft_mat = ', ft_mat.shape)
        return ft_mat

    def load_pssm_ft_mat(self):
        with open(antibody_pssm_file, 'r') as f:
            anti_pssm_dict = json.load(f)

        with open(virus_pssm_file, 'r') as f:
            virus_pssm_dict = json.load(f)

        anti_pssm_mat = np.array(list(map(lambda name: anti_pssm_dict[name], self.raw_all_antibody_name)))
        virus_pssm_mat = np.array(list(map(lambda name: virus_pssm_dict[name], self.raw_all_virus_name)))
        # print('anti_pssm_mat = ', anti_pssm_mat.shape, 'virus_pssm_mat = ', virus_pssm_mat.shape)
        return anti_pssm_mat, virus_pssm_mat

    def generate_dataset_info(self):
        self.dataset_info['kmer_dim'] = self.protein_ft_dict['virus_kmer_whole'].shape[-1]
        self.dataset_info['pssm_antibody_dim'] = self.protein_ft_dict['antibody_pssm'].shape[-1]
        self.dataset_info['pssm_virus_dim'] = self.protein_ft_dict['virus_pssm'].shape[-1]
        self.dataset_info['amino_type_num'] = max(amino_map_idx.values()) + 1
        self.dataset_info['max_antibody_len'] = self.protein_ft_dict['antibody_one_hot'].shape[1]
        self.dataset_info['max_virus_len'] = self.protein_ft_dict['virus_one_hot'].shape[1]


if __name__ == '__main__':
    dataset = AbsDataset(
        max_antibody_len=344,
        max_virus_len=912,
        train_split_param=[0.9, 0.05, 0.05],
        reprocess=True,
        kmer_min_df=0.1,
        label_norm=True)

    print(dataset.dataset_info)
    print(dataset.antibody_index_in_pair.shape)
    print('train_index ', dataset.train_index.shape)
    print('valid_seen_index ', dataset.valid_seen_index.shape)
    print('test_seen_index ', dataset.test_seen_index.shape)
    print('test_unseen_index ', dataset.test_unseen_index.shape)
