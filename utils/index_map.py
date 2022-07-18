#!/usr/bin/env
# coding:utf-8

import numpy as np

def get_pos_in_raw_arr(sub_arr, raw_arr):
    '''
    :param sub_arr:  np array shape = [m]
    :param raw_arr:  np array shape = [n] 无重复数字
    :return: np array  shape = [m]
    '''
    raw_pos_dict = {}
    for i in range(raw_arr.shape[0]):
        raw_pos_dict[raw_arr[i]] = i
    trans_pos_arr = []
    for num in sub_arr:
        trans_pos_arr.append(raw_pos_dict[num])

    return np.array(trans_pos_arr, dtype=np.long)


def get_map_index_for_sub_arr(sub_arr, raw_arr):
    map_arr = np.zeros(raw_arr.shape)
    map_arr.fill(-1)
    for idx in range(sub_arr.shape[0]):
        map_arr[sub_arr[idx]] = idx
    return map_arr


