import copy
import models
from solver import Solver
from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import os
import math
from math import isnan
import re
import pickle
import gensim
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit
from local_update import LocalUpdate

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


class FedSolver(object):
    def __init__(self, train_config, dev_config, test_config, train_data, dev_data_loader, test_data_loader, dict_users, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data = train_data
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
        self.dict_users = dict_users

    # @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(
                self.train_config)

        # Final list
        for name, param in self.model.named_parameters():

            # Bert freezing customizations
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split(
                        "encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            elif self.train_config.data == "ur_funny":
                if "bert" in name:
                    param.requires_grad = False

            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            print('\t' + name, param.requires_grad)

        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.train_config.use_bert:
            if self.train_config.pretrained_emb is not None:
                self.model.embed.weight.data = self.train_config.pretrained_emb
            self.model.embed.requires_grad = False

        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)

    @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1

        # self.criterion = criterion = nn.L1Loss(reduction="mean")
        if self.train_config.data == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else:  # mosi and mosei are regression dataloaders
            self.criterion = criterion = nn.MSELoss(reduction="mean")

        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()

        best_valid_loss = float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.5)

        for epoch in range(self.train_config.n_epoch):
            train_losses = []
            valid_losses = []
            ep_loss = []
            w_locals = []

            self.model.train()

            net_glob = self.model.state_dict()

            idxs_users = np.random.choice(
                self.train_config.clients, self.train_config.samples, replace=False)

            for idx in idxs_users:

                flag = np.random.choice(3)
                # local_train = LocalUpdate(args=self.train_config, dataloader=self.train_data[idx],dev_data_loader=self.dev_data_loader,test_data_loader=self.test_data_loader,model = copy.deepcopy(self.model))
                # local_train.build()
                # local_w, idxs_loss = local_train.train(optimizer = self.optimizer, lr_scheduler = lr_scheduler, criterion=self.criterion)

                # ep_loss.append(copy.deepcopy(idxs_loss))
                client_config = self.train_config
                client_config.n_epoch = 2
                lc_model = copy.deepcopy(self.model)
                train_data_loader = self.train_data[idx]
                local_train = Solver(
                    train_config = client_config, 
                    dev_config = None, 
                    test_config = None, 
                    train_data_loader = train_data_loader, 
                    dev_data_loader = self.dev_data_loader, 
                    test_data_loader = self.test_data_loader, 
                    is_train=True, 
                    model=lc_model,
                    flag=flag)
                local_train.build()
                local_train.train()
                local_w = local_train.model.state_dict()
                w_locals.append(copy.deepcopy(local_w))

            net_glob = self.FedAvg(w_locals)
            self.model.load_state_dict(net_glob)
            self.eval(mode="test", to_print=True)

    def FedAvg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg

    def eval(self, mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

            # if to_print:
            #     self.model.load_state_dict(torch.load(
            #         f'checkpoints/model_{self.train_config.name}.std'))

        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                y_tilde = self.model(t, v, a, l, bert_sent,
                                     bert_sent_type, bert_sent_mask)

                if self.train_config.data == "ur_funny":
                    y = y.squeeze()

                cls_loss = self.criterion(y_tilde, y)
                loss = cls_loss

                eval_loss.append(loss.item())
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)

        return eval_loss, accuracy

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """

        if self.train_config.data == "ur_funny":
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if to_print:
                print("Confusion Matrix (pos/neg) :")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report (pos/neg) :")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy (pos/neg) ",
                      accuracy_score(test_truth, test_preds))

            return accuracy_score(test_truth, test_preds)

        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array(
                [i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            # Average L1 distance between preds and truths
            mae = np.mean(np.absolute(test_preds - test_truth))
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)

            f_score = f1_score(
                (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

            # pos - neg
            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

            if to_print:
                print("mae: ", mae)
                print("corr: ", corr)
                print("mult_acc7: ", mult_a7)
                print("mult_acc5: ", mult_a5)
                print("F1_score: ", f_score)
                # print("Classification Report (pos/neg) :")
                # print(classification_report(binary_truth, binary_preds, digits=5))
                # print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))

            # # non-neg - neg
            # binary_truth = (test_truth >= 0)
            # binary_preds = (test_preds >= 0)

            # if to_print:
            #     print("Classification Report (non-neg/neg) :")
            #     print(classification_report(binary_truth, binary_preds, digits=5))
            #     print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))

            return accuracy_score(binary_truth, binary_preds)

    def get_domain_loss(self,):

        if self.train_config.use_cmd_sim:
            return 0.0

        # Predicted domain labels
        domain_pred_t = self.model.domain_label_t
        domain_pred_v = self.model.domain_label_v
        domain_pred_a = self.model.domain_label_a

        # True domain labels
        domain_true_t = to_gpu(torch.LongTensor([0]*domain_pred_t.size(0)))
        domain_true_v = to_gpu(torch.LongTensor([1]*domain_pred_v.size(0)))
        domain_true_a = to_gpu(torch.LongTensor([2]*domain_pred_a.size(0)))

        # Stack up predictions and true labels
        domain_pred = torch.cat(
            (domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat(
            (domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_cmd_loss(self,):

        if not self.train_config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_t,
                             self.model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.utt_shared_t,
                              self.model.utt_shared_a, 5)
        loss += self.loss_cmd(self.model.utt_shared_a,
                              self.model.utt_shared_v, 5)
        loss = loss/3.0

        return loss

    def get_diff_loss(self):

        shared_t = self.model.utt_shared_t
        shared_v = self.model.utt_shared_v
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_v = self.model.utt_private_v
        private_a = self.model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss

    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)
        loss += self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss/3.0
        return loss
