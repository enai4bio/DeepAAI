#!/usr/bin/env
# coding:utf-8
import sys
import os
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
from dataset.abs_dataset_reg import AbsDataset
from metrics.evaluate import evaluate_regression_v2
from dataset.dataset_torch import TorchAbsDataset, TorchAbsDatasetParapred
from models.ag_fast_parapred_reg import AgFastParapredReg
import numpy as np
import random
import torch
import torch.nn as nn
import os.path as osp


current_path = osp.dirname(osp.realpath(__file__))


class Trainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict
        self.setup_seed(self.param_dict['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = AbsDataset(max_antibody_len=self.param_dict['max_antibody_len'],
                                  max_virus_len=self.param_dict['max_virus_len'],
                                  train_split_param=self.param_dict['hot_data_split'],
                                  reprocess=True)
        self.param_dict.update(self.dataset.dataset_info)

        self.dataset.to_tensor(self.device)
        self.file_name = __file__.split('/')[-1].replace('.py', '')
        self.trainer_info = '{}_seed={}_batch={}'.format(self.file_name, self.param_dict['seed'], self.param_dict['batch_size'])
        self.save_img_path = osp.join(current_path, 'plot_curve', self.trainer_info)

        self.build_model()

    def build_model(self):
        self.model = AgFastParapredReg(
            ft_dim=self.param_dict['amino_type_num'],
            max_antibody_len=self.param_dict['max_antibody_len'],
            max_virus_len=self.param_dict['max_virus_len'],
            h_dim=self.param_dict['h_dim'],
        ).to(self.device)
        if torch.cuda.device_count() > 1:
            print("Use ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'])
        self.loss_op = nn.MSELoss(reduction='mean')
        self.best_res = None
        self.min_dif = 1e10

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
        for antibody_ft, virus_ft, pair_label, antibody_len, virus_len in dataloader:
            antibody_ft = antibody_ft.cuda()
            virus_ft = virus_ft.cuda()
            pair_label = pair_label.cuda()
            pred = self.model(
                batch_antibody_ft=antibody_ft,
                batch_virus_ft=virus_ft,
                batch_antibody_len=antibody_len,
                batch_virus_len=virus_len
            )
            pred = pred.view(-1)
            if is_training:
                r_loss = self.loss_op(pred, pair_label)
                param_l2_loss = 0
                param_l1_loss = 0
                for name, param in self.model.named_parameters():
                    if 'bias' not in name:
                        param_l2_loss += torch.norm(param, p=2)
                        param_l1_loss += torch.norm(param, p=1)

                param_l2_loss = self.param_dict['param_l2_coef'] * param_l2_loss
                loss = r_loss + param_l2_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            pred = pred.detach().to('cpu').numpy()
            pair_label = pair_label.detach().to('cpu').numpy()
            all_pred = np.hstack([all_pred, pred])
            all_label = np.hstack([all_label, pair_label])
        all_pred = all_pred.reshape(-1)
        all_label = all_label.reshape(-1)
        # evaluate
        mse, mae, mape, r2, pccs_rho, pccs_pval, spear_rho, spear_pval = \
            evaluate_regression_v2(pred=all_pred, label=all_label)
        return mse, mae, mape, r2, pccs_rho, spear_rho

    def print_res(self, res_list, epoch):
        train_mse, valid_mse, seen_test_mse, unseen_test_mse, \
        train_mae, valid_mae, seen_test_mae, unseen_test_mae,\
        train_mape, valid_mape, seen_test_mape, unseen_test_mape,\
        train_r2, valid_r2, seen_test_r2, unseen_test_r2,\
        train_pccs_rho, valid_pccs_rho, seen_test_pccs_rho, unseen_test_pccs_rho,\
        train_spear_rho, valid_spear_rho, seen_test_spear_rho, unseen_test_spear_rho = res_list

        msg_log = 'Epoch: {:03d}, ' \
                  'MSE: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'MAE: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'MAPE: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'R_SQUARED: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'pccs_rho: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'spear_rho,: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f},  ' \
            .format(epoch,
                    train_mse, valid_mse, seen_test_mse, unseen_test_mse,
                    train_mae, valid_mae, seen_test_mae, unseen_test_mae,
                    train_mape, valid_mape, seen_test_mape, unseen_test_mape,
                    train_r2, valid_r2, seen_test_r2, unseen_test_r2,
                    train_pccs_rho, valid_pccs_rho, seen_test_pccs_rho, unseen_test_pccs_rho,
                    train_spear_rho, valid_spear_rho, seen_test_spear_rho, unseen_test_spear_rho
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
            train_mse, train_mae, train_mape, train_r2, train_pccs_rho, train_spear_rho = \
                self.iteration(epoch, train_dataloader, is_training=True)

            valid_mse, valid_mae, valid_mape, valid_r2, valid_pccs_rho, valid_spear_rho = \
                self.iteration(epoch,  valid_dataloader, is_training=False)

            seen_test_mse, seen_test_mae, seen_test_mape, seen_test_r2, seen_test_pccs_rho, seen_test_spear_rho = \
                self.iteration(epoch, seen_test_dataloader, is_training=False)

            unseen_test_mse, unseen_test_mae, unseen_test_mape, unseen_test_r2, unseen_test_pccs_rho, unseen_test_spear_rho = \
                self.iteration(epoch, unseen_test_dataloader, is_training=False)

            res_list = [
                train_mse, valid_mse, seen_test_mse, unseen_test_mse,
                train_mae, valid_mae, seen_test_mae, unseen_test_mae,
                train_mape, valid_mape, seen_test_mape, unseen_test_mape,
                train_r2, valid_r2, seen_test_r2, unseen_test_r2,
                train_pccs_rho, valid_pccs_rho, seen_test_pccs_rho, unseen_test_pccs_rho,
                train_spear_rho, valid_spear_rho, seen_test_spear_rho, unseen_test_spear_rho
            ]

            if valid_mse < self.min_dif:
                self.min_dif = valid_mse
                self.best_res = res_list
                self.best_epoch = epoch
                # save model
                # same_model_param_path = osp.join(current_path, 'save_model_data', self.trainer_info + '_param.pkl')
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
            'dropout_num': 0.4,
            'lr': 5e-4,
            'param_l1_coef': 0,
            'param_l2_coef': 5e-4,
            'max_antibody_len': 344,
            'max_virus_len': 912,
            'amino_embedding_dim': 7,
            'bilstm_num_layers': 1
        }
        trainer = Trainer(**param_dict)
        trainer.start()