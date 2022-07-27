import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import torch.optim as optim
import copy

from solver import Solver
from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models
import os

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data_batch = self.dataset[self.idxs[item]]
        return data_batch


class LocalUpdate(Solver):
    def __init__(self, args, dataloader ,dev_data_loader,test_data_loader, model, dataset=None, idxs=None, is_train=True):
        self.train_config = args
        # self.train_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=1, shuffle=True)
        self.train_loader = dataloader
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.model = model
        self.is_train = is_train
    def train(self, optimizer, lr_scheduler, criterion):
        net = self.model
        self.criterion = criterion
        train_losses = []
        for e in range(2):
            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            train_loss_recon = []
            train_loss_sp = []
            train_loss = []
            for i_batch, data_batch in enumerate(self.train_loader):
                self.model.train()
                net.train()
                net.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = data_batch
                batch_size = t.size(0)
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                y_tilde = net(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                
                if self.train_config.data == "ur_funny":
                    y = y.squeeze()

                cls_loss = criterion(y_tilde, y)
                diff_loss = self.get_diff_loss()
                domain_loss = self.get_domain_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()
                
                if self.train_config.use_cmd_sim:
                    similarity_loss = cmd_loss
                else:
                    similarity_loss = domain_loss
                
                loss = cls_loss + \
                    self.train_config.diff_weight * diff_loss + \
                    self.train_config.sim_weight * similarity_loss + \
                    self.train_config.recon_weight * recon_loss

                loss.backward()
                
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)
                optimizer.step()

                train_loss_cls.append(cls_loss.item())
                train_loss_diff.append(diff_loss.item())
                train_loss_recon.append(recon_loss.item())
                train_loss.append(loss.item())
                train_loss_sim.append(similarity_loss.item())
                

            train_losses.append(np.mean(train_loss).item())
            print(f"Training loss: {round(np.mean(train_loss), 4)}")

            valid_loss, valid_acc = self.eval(mode="dev")
            print(f'Validation acc: {valid_acc}')

        return net.state_dict(), sum(train_losses)/len(train_losses)
