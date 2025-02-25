import os
import pickle
import numpy as np
from random import random
import copy

from config import get_config, activation_dict
from data_loader import get_loader, get_client_loaders
from solver import Solver
from fed_solver import FedSolver

import torch
import torch.nn as nn
from torch.nn import functional as F


if __name__ == '__main__':
    
    # Setting random seed
    random_name = str(random())
    random_seed = 336   
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    
    # Setting the config for each stage
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')
    train_config.name = 'tmpbest'
    print(train_config)

    # Creating pytorch dataloaders
    # train_data_loader = get_loader(train_config, shuffle = True)
    train_loaders, dict_users = get_client_loaders(train_config, shuffle = True)
    dev_data_loader = get_loader(dev_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = False)

    # Solver is a wrapper for model traiing and testing
    # solver = Solver
    # solver = solver(train_config, None, None, train_data_loader , dev_data_loader, test_data_loader, is_train=True)

    solver = FedSolver(train_config, dev_config, test_config, train_loaders, dev_data_loader, test_data_loader, dict_users)

    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    solver.train()
