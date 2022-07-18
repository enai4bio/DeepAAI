#!/usr/bin/env
# coding:utf-8

from sklearn import metrics as metricser
from scipy import stats

def evaluate_classification(predict_proba, label):
    '''
    :param predict_proba: float -- 0~1
    :param label: 0 or 1
    :return:
    '''
    predicts = np.around(predict_proba)
    # print(predicts)
    f1_s = metricser.f1_score(y_true=label, y_pred=predicts)
    acc = metricser.accuracy_score(y_true=label, y_pred=predicts)
    precision = metricser.precision_score(y_true=label, y_pred=predicts)
    recall = metricser.recall_score(y_true=label, y_pred=predicts)
    auc = metricser.roc_auc_score(y_true=label, y_score=predict_proba)
    # cross_entropy = bi_cross_entropy(predict_proba, label)
    # mse = bi_mse(predict_proba, label)
    # mae = bi_mae(predict_proba, label)
    mcc = mcc_score(predict_proba, label)
    return acc, precision, recall, f1_s, auc, mcc


def evaluate_classification_paper(predict_proba, label):
    '''
    :param predict_proba: float -- 0~1
    :param label: 0 or 1
    :return:
    '''
    predicts = np.around(predict_proba)
    # print(predicts)
    f1_s = metricser.f1_score(y_true=label, y_pred=predicts)
    acc = metricser.accuracy_score(y_true=label, y_pred=predicts)

    roc_auc = metricser.roc_auc_score(y_true=label, y_score=predict_proba)
    precision, recall, threshold = metricser.precision_recall_curve(y_true=label, probas_pred=predict_proba)
    pr_auc = metricser.auc(recall, precision)

    return acc, f1_s, roc_auc, pr_auc




def evaluate_classification_all(predict_proba, label):
    '''
    :param pred:
    :param label:
    :return:
    '''
    # print(predict_proba.shape, predict_proba)
    # print(label.shape, label)
    # exit()
    predicts = np.around(predict_proba)
    # print(predicts)
    f1_s = metricser.f1_score(y_true=label, y_pred=predicts)
    acc = metricser.accuracy_score(y_true=label, y_pred=predicts)
    precision = metricser.precision_score(y_true=label, y_pred=predicts)
    recall = metricser.recall_score(y_true=label, y_pred=predicts)
    auc = metricser.roc_auc_score(y_true=label, y_score=predict_proba)
    cross_entropy = bi_cross_entropy(predict_proba, label)
    mse = bi_mse(predict_proba, label)
    mae = bi_mae(predict_proba, label)
    mcc = mcc_score(predict_proba, label)
    return acc, precision, recall, f1_s, auc, mcc, cross_entropy, mae, mse




def bi_mse(predict_proba, label):
    return np.mean((predict_proba - label) ** 2)

def bi_mae(predict_proba, label):
    return np.mean(np.abs(predict_proba - label))

def bi_cross_entropy(predict_proba, label):
    # np 计算 有nan
    # cross_entropy = -(label * np.log(predict_proba) + (1 - label) * np.log(1 - predict_proba))
    # return np.mean(cross_entropy)
    # 用torch的接口  将无穷 -> 0
    predict_proba = torch.FloatTensor(predict_proba)
    label = torch.FloatTensor(label)
    return torch.nn.functional.binary_cross_entropy(predict_proba, label).numpy()

def mcc_score(predict_proba, label):
    trans_pred = np.ones(predict_proba.shape)
    trans_label = np.ones(label.shape)
    trans_pred[predict_proba < 0.5] = -1
    trans_label[label != 1] = -1
    # print(trans_pred.shape, trans_pred)
    # print(trans_label.shape, trans_label)
    mcc = metricser.matthews_corrcoef(trans_label, trans_pred)
    # mcc = metricser.matthews_corrcoef(trans_pred, trans_label)
    return mcc




import numpy as np
import torch
from sklearn.metrics import r2_score


def evaluate_regression(pred, label):
    '''
    :param pred: numpy
    :param label: numpy
    :return:
    '''
    if type(pred) == torch.Tensor:
        pred = pred.detach().to('cpu').numpy()
    if type(label) == torch.Tensor:
        label = label.detach().to('cpu').numpy()

    if pred.shape[0] <= 0 or label.shape[0] <= 0:
        return np.nan, np.nan, np.nan, np.nan
    # print(pred.shape, label.shape)
    # print('pred = ', pred)
    # print('label = ', label)
    mse = np.mean(np.square(pred - label))

    mae = np.mean(np.abs(pred - label))

    mape = np.mean(np.abs((pred - label) / (label + 1e-9)))

    r2 = r2_score(label, pred)
    return mse, mae, mape, r2

def evaluate_regression_v2(pred, label):
    '''
    :param pred: numpy
    :param label: numpy
    :return:
    '''
    try:
        if type(pred) == torch.Tensor:
            pred = pred.detach().to('cpu').numpy()
        if type(label) == torch.Tensor:
            label = label.detach().to('cpu').numpy()

        if pred.shape[0] <= 0 or label.shape[0] <= 0:
            return np.nan, np.nan, np.nan, np.nan

        mse = np.mean(np.square(pred - label))

        mae = np.mean(np.abs(pred - label))

        mape = np.mean(np.abs((pred - label) / (label + 1e-9)))

        r2 = r2_score(label, pred)

        pccs_rho, pccs_pval = stats.pearsonr(pred, label)

        spear_rho, spear_pval = stats.spearmanr(pred, label)
        return mse, mae, mape, r2, pccs_rho, pccs_pval, spear_rho, spear_pval
    except:
        print('error evaluate_regression_v2')
        return 1e5, 1e5, 1e5, 0, 0, 0, 0, 0


if __name__ == '__main__':

    y_true = np.array([1, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.2, 0.8, 0.9, 0])
    # print(mcc_score(y_score, y_true))
    print(evaluate_regression_v2(y_score, y_true))
