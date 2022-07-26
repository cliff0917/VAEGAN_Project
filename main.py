

import os
import copy

# from attr import attr
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from qqdm import qqdm, format_str
import argparse

import logging
import vaegan_models as model
from config import *


# set the random seed
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

same_seeds(2000)

parser = argparse.ArgumentParser(description="VAEGAN")
parser.add_argument("--dataset", type=str, default="AWA2")
parser.add_argument("--model_type", type=str, default="cvae")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--w_dir", type=str, default="./methods/method_test")
parser.add_argument("--n_critic", type=int, default=2)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--gzsl", type=bool, default=False)
parser.add_argument("--zsl", type=bool, default=False)
args = parser.parse_args()

# ----- set parameter
config['dataset'] = args.dataset
config['model_type'] = args.model_type
config['lr'] = args.lr
config['workspace_dir'] = args.w_dir
config['n_critic'] = args.n_critic
config['epochs'] = args.epochs
config['gzsl'] = args.gzsl
config['zsl'] = args.zsl

# best_model_path = f"vaegan_pt/{config['dataset']}/best_model_cvae.pt"
# last_model_path = f"vaegan_pt/{config['dataset']}/last_model_cvae.pt"

# set path
# global_path = "/home/csie2020/p76091543/Plearning_song/"
home_dir = os.path.expanduser("~") + '/'
global_path = home_dir + "Plearning_song/"

if config['dataset'] == 'SUN':
    config['attr_dim'] = 102
    config['latent_dim'] = 102
    config['class_num'] = 717
    config['seen_class_num'] = 645
    config['unseen_class_num'] = 72
elif config['dataset'] == 'CUB':
    config['attr_dim'] = 312
    config['latent_dim'] = 312
    config['class_num'] = 200
    config['seen_class_num'] = 150
    config['unseen_class_num'] = 50
elif config['dataset'] == 'AWA2':
    config['attr_dim'] = 85
    config['latent_dim'] = 85
    config['class_num'] = 50
    config['seen_class_num'] = 40
    config['unseen_class_num'] = 10


# Create dataset
class CustomTensorDataset(TensorDataset):
    def __init__(self, data, attr):
        self.data = data
        self.attr = attr
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.attr[index]
        return x, y

    def __len__(self):
        return len(self.data)

def get_data():
    attr_path = f"other_mats/{config['dataset']}/attr.mat"
    res_path = f"other_mats/{config['dataset']}/resnet.mat"
    mat_res = sio.loadmat(res_path)
    feature = mat_res['features'].T
    label = mat_res['labels'].astype(int).squeeze() - 1
    mat_attr = sio.loadmat(attr_path)
    trainval_loc = mat_attr['trainval_loc'].squeeze() - 1
    
    train_feature = torch.from_numpy(feature).float().to(device)
    attribute = torch.from_numpy(mat_attr['att'].T).float().to(device)
    attribute = attribute[label]
    return train_feature, attribute

x, y = get_data()

train_dataset = CustomTensorDataset(x, y)
train_sampler = RandomSampler(train_dataset)
train_data_loader = DataLoader(train_dataset, sampler=train_sampler, 
                            batch_size=config["batch_size"])
config['dataloader'] = train_data_loader


criterion = nn.MSELoss(reduction='mean')
# optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)


# model train
trainer = model.TrainerGAN(config)

if config['model_type'] == 'cvae':
    print("train cvae")
    trainer.train_cvae()
elif config['model_type'] == 'vae':
    print("train vae")
    trainer.train_vae()
elif config['model_type'] == 'cvaegan' or config['model_type'] == 'vaegan':
    print("trainer.train()")
    trainer.train()
else:
    print("trainer.train_cycle_vaegan()")
    trainer.train_cycle_vaegan()





