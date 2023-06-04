import numpy as np
from feature_trans_content import amino_physicochemical_ft, amino_pssm_ft, amino_map_idx
from sklearn.preprocessing import StandardScaler
import copy

def load_stand_physicochemical_ft():
    '''
    :return: dict key: str_amino  value: np_array
    '''
    ft_mat = []
    amino_names = list(amino_physicochemical_ft.keys())
    for key in amino_names:
        ft_mat.append(amino_physicochemical_ft[key])
    stand_scaler = StandardScaler()
    stand_scaler.fit(ft_mat)
    ft_mat = stand_scaler.transform(ft_mat)

    stand_amino_ft_dict = {}
    for idx in range(len(amino_names)):
        key = amino_names[idx]
        value = ft_mat[idx]
        stand_amino_ft_dict[key] = value

    return stand_amino_ft_dict


def get_padding_pssm_ft():
    ft_mat = []
    amino_names = list(amino_pssm_ft.keys())
    for key in amino_names:
        ft_mat.append(amino_pssm_ft[key])
    return np.mean(ft_mat, axis=0)


def get_padding_ft_dict():
    amino_one_hot_ft_pad_dict = {}
    pad_amino_map_idx = copy.deepcopy(amino_map_idx)
    padding_num = max(pad_amino_map_idx.values()) + 1 # 0-21, padding
    amino_ft_dim = padding_num + 1
    for atom_name in pad_amino_map_idx.keys():
        ft = np.zeros(amino_ft_dim)
        ft[pad_amino_map_idx[atom_name]] = 1
        amino_one_hot_ft_pad_dict[atom_name] = ft

    pad_amino_map_idx['pad'] = padding_num
    padding_ft = np.zeros(amino_ft_dim)
    padding_ft.fill(1/amino_ft_dim)
    amino_one_hot_ft_pad_dict['pad'] = padding_ft

    amino_pssm_ft_pad_dict = copy.copy(amino_pssm_ft)
    amino_pssm_ft_pad_dict['pad'] = get_padding_pssm_ft()

    amino_physicochemical_ft_pad_dict = load_stand_physicochemical_ft()
    physicochemical_ft_size = len(list(amino_physicochemical_ft.values())[0])
    amino_physicochemical_ft_pad_dict['pad'] = np.zeros(physicochemical_ft_size)

    return amino_one_hot_ft_pad_dict, amino_pssm_ft_pad_dict, amino_physicochemical_ft_pad_dict, pad_amino_map_idx


def get_index_in_target_list(need_trans_list, target_list):
    trans_index = []
    for need_trans_str in need_trans_list:
        idx = target_list.index(need_trans_str)
        trans_index.append(idx)
    return np.array(trans_index)

def get_target_type_idx_in_all_name(target_name, raw_list):
    target_index = []
    for idx in range(len(raw_list)):
        if target_name in raw_list[idx]:
            target_index.append(idx)
    return np.array(target_index)



def generate_ac_feature(protein_ft, max_lag=30):
    '''
    :param protein_ft: len*amino_dim
    :param max_lag: num  (1~max_len)
    :return:  ac_ft [amino_dim * lag]
    '''

    protein_len = protein_ft.shape[0]

    mean_amino_ft = np.mean(protein_ft, axis=0)
    ft_mat = []
    for lag in range(max_lag):
        tmp_ft = []
        for idx in range(protein_len - lag):
            co_variance = (protein_ft[idx, :] - mean_amino_ft) * (protein_ft[idx+lag, :] - mean_amino_ft)
            tmp_ft.append(co_variance)
        tmp_ft = np.mean(np.array(tmp_ft), axis=0)
        ft_mat.append(tmp_ft)
    return np.array(ft_mat)


def get_all_protein_ac_feature(protein_ft_list, max_lag=30):
    ac_ft_mat = []
    for protein_ft in protein_ft_list:
        ac_ft_mat.append(generate_ac_feature(protein_ft, max_lag=max_lag))
    return np.array(ac_ft_mat)

if __name__ == '__main__':
    get_padding_ft_dict()