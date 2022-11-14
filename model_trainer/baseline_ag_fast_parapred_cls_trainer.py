#!/usr/bin/env
# coding:utf-8
import sys
import os
sys.path.append(os.getcwd())

import torch.nn as nn
from dataset.abs_dataset_cls import AbsDataset
from metrics.evaluate import evaluate_classification
from dataset.dataset_torch import TorchAbsDataset, TorchAbsDatasetParapred
from models.ag_fast_parapred_cls import AgFastParapredCls
import numpy as np
import random
import torch
import os.path as osp
from torch.utils.data import DataLoader

current_path = osp.dirname(osp.realpath(__file__))
save_model_data_dir = 'save_model_param_pred'


class Trainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict
        self.setup_seed(self.param_dict['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = AbsDataset(max_antibody_len=self.param_dict['max_antibody_len'],
                                  max_virus_len=self.param_dict['max_virus_len'],
                                  train_split_param=self.param_dict['hot_data_split'],
                                  label_type=self.param_dict['label_type'],
                                  reprocess=True)
        self.param_dict.update(self.dataset.dataset_info)

        self.dataset.to_tensor(self.device)
        self.file_name = __file__.split('/')[-1].replace('.py', '')
        self.trainer_info = '{}_seed={}_batch={}'.format(self.file_name, self.param_dict['seed'], self.param_dict['batch_size'])
        self.build_model()

    def build_model(self):
        self.model = AgFastParapredCls(
            ft_dim=self.param_dict['amino_type_num'],
            max_antibody_len=self.param_dict['max_antibody_len'],
            max_virus_len=self.param_dict['max_virus_len'],
            h_dim=self.param_dict['h_dim'],
        ).to(self.device)
        if torch.cuda.device_count() > 1:
            print("Use ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'])
        self.loss_op = nn.BCELoss()
        self.best_res = None
        self.min_dif = -1e10

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def iteration(self, epoch, dataloader, is_training=False):
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        all_pred = []
        all_label = []
        all_loss = []
        for antibody_ft, virus_ft, pair_label, antibody_len, virus_len in dataloader:
            antibody_ft = antibody_ft.cuda()
            virus_ft = virus_ft.cuda()
            pair_label = pair_label.float().cuda()

            pred = self.model(
                batch_antibody_ft=antibody_ft,
                batch_virus_ft=virus_ft,
                batch_antibody_len=antibody_len,
                batch_virus_len=virus_len
            )
            pred = pred.view(-1)
            if is_training:
                c_loss = self.loss_op(pred, pair_label)
                param_l2_loss = 0
                param_l1_loss = 0
                for name, param in self.model.named_parameters():
                    if 'bias' not in name:
                        param_l2_loss += torch.norm(param, p=2)
                        param_l1_loss += torch.norm(param, p=1)

                param_l2_loss = self.param_dict['param_l2_coef'] * param_l2_loss
                loss = c_loss + param_l2_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                all_loss.append(loss.detach().to('cpu').item())

            pred = pred.detach().to('cpu').numpy()
            pair_label = pair_label.detach().to('cpu').numpy()
            all_pred = np.hstack([all_pred, pred])
            all_label = np.hstack([all_label, pair_label])

        return all_pred, all_label, all_loss

    def print_res(self, res_list, epoch):
        train_acc, seen_valid_acc, seen_test_acc, unseen_test_acc, \
        train_p, seen_valid_p, seen_test_p, unseen_test_p, \
        train_r, seen_valid_r, seen_test_r, unseen_test_r, \
        train_f1, seen_valid_f1, seen_test_f1, unseen_test_f1, \
        train_auc, seen_valid_auc, seen_test_auc, unseen_test_auc, \
        train_mcc, seen_valid_mcc, seen_test_mcc, unseen_test_mcc = res_list

        msg_log = 'Epoch: {:03d}, ' \
                  'ACC: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'P: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'R: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'F1: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'AUC: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'MCC: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f},  ' \
            .format(epoch,
                    train_acc, seen_valid_acc, seen_test_acc, unseen_test_acc,
                    train_p, seen_valid_p, seen_test_p, unseen_test_p, \
                    train_r, seen_valid_r, seen_test_r, unseen_test_r, \
                    train_f1, seen_valid_f1, seen_test_f1, unseen_test_f1, \
                    train_auc, seen_valid_auc, seen_test_auc, unseen_test_auc, \
                    train_mcc, seen_valid_mcc, seen_test_mcc, unseen_test_mcc
                    )
        print(msg_log)

    def start(self, display=True):
        train_dataset = TorchAbsDatasetParapred(self.dataset, split_type='train', ft_type='one_hot')
        train_dataloader = DataLoader(train_dataset, batch_size=self.param_dict['batch_size'], shuffle=True)

        valid_dataset = TorchAbsDatasetParapred(self.dataset, split_type='valid', ft_type='one_hot')
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.param_dict['batch_size'], shuffle=True)

        seen_test_dataset = TorchAbsDatasetParapred(self.dataset, split_type='seen_test', ft_type='one_hot')
        seen_test_dataloader = DataLoader(seen_test_dataset, batch_size=self.param_dict['batch_size'], shuffle=True)

        unseen_test_dataset = TorchAbsDatasetParapred(self.dataset, split_type='unseen_test', ft_type='one_hot')
        unseen_test_dataloader = DataLoader(unseen_test_dataset, batch_size=self.param_dict['batch_size'], shuffle=True)

        for epoch in range(1, self.param_dict['epoch_num'] + 1):
            # train
            train_pred, train_label, train_loss = self.iteration(epoch, train_dataloader, is_training=True)
            train_acc, train_p, train_r, train_f1, train_auc, train_mcc = \
                evaluate_classification(predict_proba=train_pred, label=train_label)

            valid_pred, valid_label, valid_loss = self.iteration(epoch,  valid_dataloader, is_training=False)
            seen_valid_acc, seen_valid_p, seen_valid_r, seen_valid_f1, seen_valid_auc, seen_valid_mcc = \
                evaluate_classification(predict_proba=valid_pred, label=valid_label)

            seen_test_pred, seen_test_label, seen_test_loss = self.iteration(epoch, seen_test_dataloader, is_training=False)
            seen_test_acc, seen_test_p, seen_test_r, seen_test_f1, seen_test_auc, seen_test_mcc = \
                evaluate_classification(predict_proba=seen_test_pred, label=seen_test_label)

            unseen_test_pred, unseen_test_label, unseen_test_loss = self.iteration(epoch, unseen_test_dataloader, is_training=False)
            unseen_test_acc, unseen_test_p, unseen_test_r, unseen_test_f1, unseen_test_auc, unseen_test_mcc = \
                evaluate_classification(predict_proba=unseen_test_pred, label=unseen_test_label)

            res_list = [
                train_acc, seen_valid_acc, seen_test_acc, unseen_test_acc,
                train_p, seen_valid_p, seen_test_p, unseen_test_p,
                train_r, seen_valid_r, seen_test_r, unseen_test_r,
                train_f1, seen_valid_f1, seen_test_f1, unseen_test_f1,
                train_auc, seen_valid_auc, seen_test_auc, unseen_test_auc,
                train_mcc, seen_valid_mcc, seen_test_mcc, unseen_test_mcc
            ]

            if seen_valid_acc > self.min_dif:
                self.min_dif = seen_valid_acc
                self.best_res = res_list
                self.best_epoch = epoch
                # save model
                # same_model_param_path = osp.join(current_path, '..', save_model_data_dir, self.trainer_info + '_param.pkl')
                # torch.save(self.model.state_dict(), same_model_param_path)

            if display:
                self.print_res(res_list, epoch)

            if epoch % 50 == 0 and epoch > 0:
                print('Best res')
                self.print_res(self.best_res, self.best_epoch)


if __name__ == '__main__':
    for seed in range(20):
        print('seed = ', seed)
        param_dict = {
            'hot_data_split': [0.9, 0.05, 0.05],
            'seed': seed,
            'batch_size': 32,
            'epoch_num': 200,
            'h_dim': 512,
            'lr': 5e-4,
            'param_l1_coef': 0,
            'param_l2_coef': 5e-4,
            'max_antibody_len': 344,
            'max_virus_len': 912,
            'label_type': 'label_10',
        }
        trainer = Trainer(**param_dict)
        trainer.start()