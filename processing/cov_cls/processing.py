import os.path as osp
import numpy as np
import pandas as pd
import json
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

from dataset_tools import get_padding_ft_dict, get_index_in_target_list, get_all_protein_ac_feature
from dataset_split import train_test_split
from k_mer_utils import k_mer_ft_generate, KmerTranslator

current_path = osp.dirname(osp.realpath(__file__))
corpus_dir = 'corpus/cov_cls'
processed_dir = 'corpus/processed_mat'
pssm_dir = 'pssm'

antibody_pssm_file = osp.join(current_path, pssm_dir, "anti_medp_pssm_840.json")
virus_pssm_file = osp.join(current_path, pssm_dir, "virus_medp_pssm_420.json")
data_file = osp.join(current_path, 'dataset_cov_cls.xlsx')

dataset_name = 'abs_dataset_cov_cls'
max_antibody_len= 436
max_virus_len = 1288
kmer_min_df = 0.1
dataset_param_str = '{}_antibody={}_virus={}_kmer_min_df={}'.format(
            dataset_name, max_antibody_len, max_virus_len, kmer_min_df)
protein_ft_save_path = osp.join(current_path, processed_dir, dataset_param_str+'__protein_ft_dict.pkl')

amino_one_hot_ft_pad_dict, amino_pssm_ft_pad_dict, amino_physicochemical_ft_pad_dict, amino_map_idx = get_padding_ft_dict()

def processing():
    data_df = pd.read_excel(data_file)

    split = data_df['split'].to_numpy()
    train_index = np.where(split == 'seen')[0]
    test_unseen_index = np.where(split == 'unseen')[0]
    all_label_mat = data_df['label'].to_numpy().astype(np.long)

    raw_all_antibody_seq_list = data_df['antibody_seq'].to_list()
    raw_all_virus_seq_list = data_df['virus_seq'].to_list()

    raw_all_antibody_set = list(sorted(set(raw_all_antibody_seq_list)))
    raw_all_virus_set = list(sorted(set(raw_all_virus_seq_list)))

    raw_all_antibody_set_len = np.array(
            list(map(lambda x: len(x), raw_all_antibody_set)))
    raw_all_virus_set_len = np.array(
            list(map(lambda x: len(x), raw_all_virus_set)))

    antibody_index_in_pair = get_index_in_target_list(raw_all_antibody_seq_list, raw_all_antibody_set)
    virus_index_in_pair = get_index_in_target_list(raw_all_virus_seq_list, raw_all_virus_set)

    known_antibody_idx = np.unique(antibody_index_in_pair)
    # unknown_antibody_idx = np.unique(antibody_index_in_pair[test_unseen_index])
    known_virus_idx = np.unique(virus_index_in_pair)

    protein_ft_dict = {}

    # one-hot
    protein_ft_dict['antibody_one_hot'] = protein_seq_list_to_ft_mat(
        raw_all_antibody_set, max_antibody_len, ft_type='amino_one_hot')
    protein_ft_dict['virus_one_hot'] = protein_seq_list_to_ft_mat(
        raw_all_virus_set, max_virus_len, ft_type='amino_one_hot')

    # # pssm
    # protein_ft_dict['antibody_pssm'], protein_ft_dict['virus_pssm'] = load_pssm_ft_mat()

    # amino_num
    protein_ft_dict['antibody_amino_num'] = protein_seq_list_to_ft_mat(
        raw_all_antibody_set, max_antibody_len, ft_type='amino_num')
    protein_ft_dict['virus_amino_num'] = protein_seq_list_to_ft_mat(
        raw_all_virus_set, max_virus_len, ft_type='amino_num')

    # k-mer-whole
    kmer_translator = KmerTranslator(trans_type='std', min_df=kmer_min_df, name=dataset_param_str)
    protein_ft = kmer_translator.fit_transform(raw_all_antibody_set + raw_all_virus_set)
    # kmer_translator.save()
    protein_ft_dict['antibody_kmer_whole'] = protein_ft[0: len(raw_all_antibody_set)]
    protein_ft_dict['virus_kmer_whole'] = protein_ft[len(raw_all_antibody_set):]

    # save
    np.save(osp.join(current_path, corpus_dir, 'train_index'), train_index)
    np.save(osp.join(current_path, corpus_dir, 'test_unseen_index'), test_unseen_index)
    np.save(osp.join(current_path, corpus_dir, 'all_label_mat'), all_label_mat)
    np.save(osp.join(current_path, corpus_dir, 'antibody_index_in_pair'), antibody_index_in_pair)
    np.save(osp.join(current_path, corpus_dir, 'virus_index_in_pair'), virus_index_in_pair)
    np.save(osp.join(current_path, corpus_dir, 'known_antibody_idx'), known_antibody_idx)
    # np.save(osp.join(current_path, corpus_dir, 'unknown_antibody_idx'), unknown_antibody_idx)
    np.save(osp.join(current_path, corpus_dir, 'known_virus_idx'), known_virus_idx)
    np.save(osp.join(current_path, corpus_dir, 'raw_all_antibody_set_len'), raw_all_antibody_set_len)
    np.save(osp.join(current_path, corpus_dir, 'raw_all_virus_set_len'), raw_all_virus_set_len)
    with open(protein_ft_save_path, 'wb') as f:
        if sys.version_info > (3, 0):
            pickle.dump(protein_ft_dict, f)
        else:
            pickle.dump(protein_ft_dict, f)

def protein_seq_list_to_ft_mat(protein_seq_list, max_sql_len, ft_type='amino_one_hot'):
    ft_mat = []
    for protein_seq in protein_seq_list:
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

# def load_pssm_ft_mat():
#     with open(antibody_pssm_file, 'r') as f:
#         anti_pssm_dict = json.load(f)

#     with open(virus_pssm_file, 'r') as f:
#         virus_pssm_dict = json.load(f)

#     anti_pssm_mat = np.array(list(map(lambda name: anti_pssm_dict[name], raw_all_antibody_name)))
#     virus_pssm_mat = np.array(list(map(lambda name: virus_pssm_dict[name], raw_all_virus_name)))
#     return anti_pssm_mat, virus_pssm_mat

if __name__ == '__main__':
    processing()