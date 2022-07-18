#!/usr/bin/env
# coding:utf-8

import numpy as np


def sort_by_select_type(name_list, type_list):
    set_of_type = list(sorted(set(type_list)))
    name_list = np.array(name_list)
    type_list = np.array(type_list)

    sorted_name_set = []
    for type_name in set_of_type:
        sorted_name_set = np.hstack((sorted_name_set,
                   sorted(name_list[type_list == type_name])
                   ))
    return list(sorted_name_set)


def set_sort_by_select_type(name_list, type_list):
    set_of_name_list = list(set(name_list))
    # print('set_sort_by_select_type = ', len(name_list), len(set_of_name_list))
    set_of_type_list = []
    for name in set_of_name_list:
        type_str = type_list[name_list.index(name)]
        set_of_type_list.append(type_str)

    return sort_by_select_type(set_of_name_list, set_of_type_list)
