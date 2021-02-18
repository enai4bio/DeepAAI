#!/usr/bin/env
# coding:utf-8
"""
Created on 2020/6/4 15:31

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
import torch.nn.functional as F


def get_degree_mat(adj_mat, pow=1, degree_version='v1'):
    degree_mat = torch.eye(adj_mat.size()[0]).to(adj_mat.device)

    if degree_version == 'v1':
        degree_list = torch.sum((adj_mat > 0), dim=1).float()
    elif degree_version == 'v2':
        # adj_mat_hat = adj_mat.data
        # adj_mat_hat[adj_mat_hat < 0] = 0
        adj_mat_hat = F.relu(adj_mat)
        degree_list = torch.sum(adj_mat_hat, dim=1).float()
    elif degree_version == 'v3':
        degree_list = torch.sum(adj_mat, dim=1).float()
        degree_list = F.relu(degree_list)
    else:
        exit('error degree_version ' + degree_version)
    degree_list = torch.pow(degree_list, pow)
    degree_mat = degree_mat * degree_list
    # degree_mat = torch.pow(degree_mat, pow)
    # degree_mat[degree_mat == float("Inf")] = 0
    # degree_mat.requires_grad = False
    # print('degree_mat = ', degree_mat)
    return degree_mat


def get_laplace_mat(adj_mat, type='sym', add_i=False, degree_version='v2'):
    if type == 'sym':
        # Symmetric normalized Laplacian
        if add_i is True:
            adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        else:
            adj_mat_hat = adj_mat
        # adj_mat_hat = adj_mat_hat[adj_mat_hat > 0]
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-0.5, degree_version=degree_version)
        # print(degree_mat_hat.dtype, adj_mat_hat.dtype)
        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        # print(laplace_mat)
        laplace_mat = torch.mm(laplace_mat, degree_mat_hat)
        return laplace_mat
    elif type == 'rw':
        # Random walk normalized Laplacian
        adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-1)
        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        return laplace_mat


def adj_2_bias(adj):
    node_size = adj.size()[0]
    new_mat = ((torch.eye(node_size).to(adj.device) + adj) >= 1).float()
    # print(new_mat)
    new_mat = torch.tensor(-1e9) * (1 - new_mat)
    # print(new_mat)
    return new_mat


def adj_2_bias_without_self_loop(adj):
    node_size = adj.size()[0]
    new_mat = (adj >= 1).float()
    # print(new_mat)
    new_mat = torch.tensor(-1e9).to(adj.device) * (1 - new_mat)
    # print(new_mat)
    return new_mat


import torch.nn as nn
from math import ceil

def fusion_conv1d_for_vec(conv_input, conv_param, kernel_size=3, stride=1, padding=0):
    '''
    :param conv_input:  batch_size * input_dim
    :param conv_param:  batch_size * param_dim
    :return: res: batch_size * out_dim,    out_dim = ceil(param_dim / kernel_size) * ((input_dim - kernel_size + padding) / stride + 1)
    '''
    # print('conv_input.size(), conv_param.size() = ', conv_input.size(), conv_param.size())
    batch_size = conv_input.size()[0]
    param_vec_dim = conv_param.size()[1]

    # conv_param to  batch_size, channel, kernel_size
    # param_vec kernel_size 如果不能被整除 padding 0
    n_channel = ceil(param_vec_dim / kernel_size)

    padding_num = n_channel * kernel_size - param_vec_dim
    param_pad_conv = nn.ConstantPad1d((0, padding_num), 0)
    conv_param = param_pad_conv(conv_param).view(batch_size, n_channel, kernel_size)
    conv_param = torch.softmax(conv_param, dim=-1)
    # print('conv_param size = ', conv_param.size())

    out_mat = []
    for batch_idx in range(batch_size):

        input_mat = conv_input[batch_idx].view(1, 1, -1)
        param_mat = conv_param[batch_idx, :, :].view(n_channel, 1, kernel_size)
        conv_res = F.conv1d(input_mat, param_mat, stride=stride, padding=padding)
        # print(batch_idx, conv_res.size())
        out_mat.append(conv_res)
    out_mat = torch.cat(out_mat, dim=0).view(batch_size, -1)
    # print(out_mat.size())
    return out_mat


def fusion_conv1d_for_mat(conv_input, conv_param, kernel_width=7, stride=2):
    '''
    :param conv_input:   batch * h * input_dim
    :param conv_param:   batch * h * paran_dim
    :param kernel_width:  kernel_size ->  h * kernel_width
    :param stride:
    :return:
    '''
    batch_size = conv_input.size()[0]
    param_mat_w = conv_param.size()[2]
    kernel_height = conv_param.size()[1]

    conv_input = conv_input.view(batch_size, 1, kernel_height, -1)
    # conv_param = conv_param.view(batch_size, 1, param_mat_w, -1)

    n_channel = ceil(param_mat_w / kernel_width)
    padding_num = n_channel * kernel_width - param_mat_w
    param_pad_conv = nn.ConstantPad2d((0, padding_num, 0, 0), 0)
    conv_param = param_pad_conv(conv_param).view(batch_size, kernel_height, n_channel, kernel_width)
    conv_param = conv_param.transpose(1, 2)

    # print(conv_param.size())

    out_mat = []
    for batch_idx in range(batch_size):
        input_mat = conv_input[batch_idx].view(1, 1, kernel_height, -1)
        param_mat = conv_param[batch_idx, :, :].view(n_channel, 1, kernel_height, kernel_width)
        conv_res = F.conv2d(input_mat, param_mat, stride=stride)
        # print(batch_idx, conv_res.size())
        out_mat.append(conv_res)
    out_mat = torch.cat(out_mat, dim=0)
    # print(out_mat.size())

    return out_mat


def simplified_lstm(lstm_layer, input, h_0=None, c_0=None):
    if lstm_layer.bidirectional:
        num_directions = 2
    else:
        num_directions = 1

    if h_0 is None:
        h_0 = torch.randn(num_directions * lstm_layer.num_layers, input.size()[0], lstm_layer.hidden_size).to(input.device)
    if c_0 is None:
        c_0 = torch.randn(num_directions * lstm_layer.num_layers, input.size()[0], lstm_layer.hidden_size).to(input.device)

    output, (hn, cn) = lstm_layer(input, (h_0, c_0))
    return output, (hn, cn)
