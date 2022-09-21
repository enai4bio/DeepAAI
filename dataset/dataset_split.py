#!/usr/bin/env
# coding:utf-8

import numpy as np


def train_test_split(record_num, split_param, mode='ratio', shuffle=False, category=None, extra_to='train'):
    '''
    :param record_num:
    :param split_param: The proportion or number of data divided
    :param mode: 'ratio' or 'numerical'
    :param shuffle:
    :param category: list record_num
    :return:  each category has the same ratio of train/valid/test
    '''

    if category is not None:
        set_of_category = set(category)
        set_of_category = sorted(set_of_category)
        # print(set_of_category)
        # print(set_of_category, count_num_of_category)

        # 每一种类别 对应的idx

        all_idx = np.arange(record_num)
        category = np.array(category)
        category_idx_dict = {}
        # print(len(category))
        # print('all c', type(category), category)
        for c in set_of_category:
            category_idx_dict[c] = all_idx[c == category]
        # 对每一种类别 分别进行划分
        train_index = []
        valid_index = []
        test_index = []

        for c in category_idx_dict:
            index_arr = category_idx_dict[c]
            sub_train_index, sub_valid_index, sub_test_index = arr_split(
                index_arr, split_param=split_param, mode=mode, shuffle=shuffle, extra_to=extra_to)
            # print(c, sub_train_index, sub_valid_index, sub_test_index)
            train_index.append(sub_train_index)
            valid_index.append(sub_valid_index)
            test_index.append(sub_test_index)
        train_index = np.concatenate(train_index, axis=0)
        valid_index = np.concatenate(valid_index, axis=0)
        test_index = np.concatenate(test_index, axis=0)
        # print('train_index = ', train_index)
    else:
        index_arr = np.arange(record_num)
        # print('index_arr =')
        train_index, valid_index, test_index = arr_split(
                index_arr, split_param=split_param, mode=mode, shuffle=shuffle, extra_to=extra_to)

    return train_index, valid_index, test_index


def arr_split(raw_arr, split_param, mode='ratio', shuffle=False, extra_to='train'):
    '''
    :param raw_arr:  index np arr
    :param split_param:
    :param mode:
    :param shuffle:
    :return:  train/valid/test index arr
    '''
    record_num = raw_arr.shape[0]
    if mode == 'ratio' and sum(split_param) <= 1:
        split_param = np.array(split_param)
        train_size, valid_size, test_size = map(int, np.floor(split_param * record_num))
        extra_size = record_num - train_size - valid_size - test_size
        # print(train_size, valid_size, test_size, extra_size, extra_to)
        if extra_to == 'train':
            train_size += extra_size
        if extra_to == 'valid':
            valid_size += extra_size
        if extra_to == 'test':
            test_size += extra_size
        # print(train_size, valid_size, test_size, extra_size)
    elif mode == 'numerical' or sum(split_param) > 1:
        split_param = np.array(split_param)
        # print(np.sum(split_param), node_size)
        assert np.sum(split_param) <= record_num
        train_size, valid_size, test_size = split_param

    node_index = np.arange(record_num)
    if shuffle is True:
        np.random.shuffle(node_index)

    train_pos = node_index[0:train_size]
    valid_pos = node_index[train_size: valid_size + train_size]
    test_pos = node_index[valid_size + train_size: valid_size + train_size + test_size]

    return raw_arr[train_pos], raw_arr[valid_pos], raw_arr[test_pos]


if __name__ == '__main__':
    # demo
    c = ['a','a','b','b','a','b','b','a','b','c','c','a','c','c','a','c','a','b','b']
    # print(len(c))
    import random
    random.seed(0)
    np.random.seed(0)

    print(train_test_split(19, [0.9, 0.1, 0], category=None, extra_to='test'))

    # raw_arr = np.arange(20)
    # print(arr_split(raw_arr, [0.6, 0.2, 0.2], shuffle=True))