#!/usr/bin/env
# coding:utf-8



import os.path as osp
import numpy as np
import pandas as pd
import torch
import json
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

from dataset.dataset_tools import get_padding_ft_dict, get_index_in_target_list
from dataset.dataset_split import train_test_split
from dataset.k_mer_utils import KmerTranslator

amino_one_hot_ft_pad_dict, amino_physicochemical_ft_pad_dict, amino_map_idx = get_padding_ft_dict()
current_path = osp.dirname(osp.realpath(__file__))
processed_dir = 'processed_mat'

antibody_pssm_file = osp.join(current_path, 'corpus', 'hiv_pssm', "anti_medp_pssm_840.json")
virus_pssm_file = osp.join(current_path, 'corpus', 'hiv_pssm', "virus_medp_pssm_420.json")
data_file = osp.join(current_path, 'corpus', 'classification_data', 'cls_data.csv')


class AbsDataset():
    def __init__(self, max_antibody_len=344, max_virus_len=912,
                 train_split_param=[0.9, 0.05, 0.05],
                 reprocess=False,
                 kmer_min_df=0.1,
                 label_type='label_10'
                 ):
        assert label_type in ['label_10', 'label_50']
        self.dataset_name = __file__.split('/')[-1].replace('.py', '')
        self.dataset_param_str = '{}_antibody={}_virus={}_kmer_min_df={}'.format(
            self.dataset_name, max_antibody_len, max_virus_len, kmer_min_df)
        self.protein_ft_save_path = osp.join(current_path, processed_dir, self.dataset_param_str+'__protein_ft_dict.pkl')
        self.reporcess = reprocess
        self.train_split_param = train_split_param
        self.label_type = label_type
        self.kmer_min_df = kmer_min_df
        self.max_antibody_len = max_antibody_len
        self.max_virus_len = max_virus_len

        self.dataset_info = {}
        self.protein_ft_dict = {}
        # key: (antibody_heavy or antibody_light or virus)_(ft: one_hot or pssm or k_mer)

        self.all_label_mat = None  # 1: ic50<10,  0: ic50>=10
        self.all_ic50 = None

        self.raw_all_antibody_name = None
        self.raw_all_virus_name = None
        self.raw_all_antibody_id = None
        self.raw_all_virus_id = None

        self.raw_all_antibody_seq_list = None
        self.raw_all_virus_seq_list = None
        self.raw_all_antibody_set = None
        self.raw_all_virus_set = None

        self.raw_all_antibody_set_len = None
        self.raw_all_virus_set_len = None

        self.antibody_index_in_pair = None
        self.virus_index_in_pair = None

        self.known_antibody_idx = None
        self.known_virus_idx = None
        self.unknown_antibody_idx = None

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

    def load_file(self):
        data_df = pd.read_csv(data_file, index_col=0)

        jr_seen_split = data_df['cls_split'].to_numpy()
        self.train_index = np.where(jr_seen_split == 'seen')[0]
        self.test_unseen_index = np.where(jr_seen_split == 'unseen')[0]

        if self.label_type == 'label_10':
            self.all_label_mat = data_df['label_10'].to_numpy().astype(np.long)
        else:
            self.all_label_mat = data_df['label_50'].to_numpy().astype(np.long)

        self.raw_all_antibody_seq_list = data_df['antibody_seq'].to_list()
        self.raw_all_virus_seq_list = data_df['virus_seq'].to_list()

        # Different pairs may have the same sequence(antibody/virus)
        protein_seq_map_to_name = {}
        protein_seq_map_to_id = {}
        for idx, row in data_df.iterrows():
            protein_seq_map_to_id[row['antibody_seq']] = row['antibody_id']
            protein_seq_map_to_id[row['virus_seq']] = row['virus_id']
            protein_seq_map_to_name[row['antibody_seq']] = row['antibody_name']
            protein_seq_map_to_name[row['virus_seq']] = row['virus_name']

        self.raw_all_antibody_set = list(sorted(set(self.raw_all_antibody_seq_list)))
        self.raw_all_virus_set = list(sorted(set(self.raw_all_virus_seq_list)))

        self.raw_all_antibody_name = list(map(lambda s: protein_seq_map_to_name[s], self.raw_all_antibody_set))
        self.raw_all_virus_name = list(map(lambda s: protein_seq_map_to_name[s], self.raw_all_virus_set))
        self.raw_all_antibody_id = list(map(lambda s: protein_seq_map_to_id[s], self.raw_all_antibody_set))
        self.raw_all_virus_id = list(map(lambda s: protein_seq_map_to_id[s], self.raw_all_virus_set))

        # Count the length of each sequence
        self.raw_all_antibody_set_len = np.array(
            list(map(lambda x: len(x), self.raw_all_antibody_set)))
        self.raw_all_virus_set_len = np.array(
            list(map(lambda x: len(x), self.raw_all_virus_set)))

        self.antibody_index_in_pair = get_index_in_target_list(self.raw_all_antibody_seq_list, self.raw_all_antibody_set)
        self.virus_index_in_pair = get_index_in_target_list(self.raw_all_virus_seq_list, self.raw_all_virus_set)

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
        kmer_translator.save()
        self.protein_ft_dict['antibody_kmer_whole'] = protein_ft[0: len(self.raw_all_antibody_set)]
        self.protein_ft_dict['virus_kmer_whole'] = protein_ft[len(self.raw_all_antibody_set):]

        # SAVE
        print('save protein ft dict ', self.protein_ft_save_path)
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
            # cat amino ft
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
        # Reading data back
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
        kmer_min_df=0.1,  # 1893
        train_split_param=[0.9, 0.05, 0.05],
        reprocess=False,
        label_type='label_10'  # (array([0, 1]), array([14792, 14602]))
    )
    print(dataset.dataset_info)
    print(dataset.raw_all_antibody_id, type(dataset.raw_all_antibody_id))
    print(dataset.raw_all_antibody_name)

