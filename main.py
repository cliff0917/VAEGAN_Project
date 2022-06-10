

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

# os.environ["CUDA_VISIBLE_DEVICES"]=""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Train config
config = {
        'num_shots': 0,
        'device': device,
        'generalized': True,
        'batch_size' : 128,
        'dataset' : 'AWA2',
        'latent_dim': 85,               # size of latent code
        'attr_dim' : 85,                # size of attr
        'nz': 85,                       # size of the latent z vector
        'class_num' : 50,               # number of total classes
        'seen_class_num' : 40,          # number of seen classes
        'unseen_class_num' : 10,        # number of unseen classes
        'd_hdim': 2048,                 # size of the hidden units in discriminator
        'visual_dim': 2048,             # size of the visual feature
        'enc_hidden_dims' : [2048, 1024],   # hidden dims of encoder
        'dec_hidden_dims' : [1024, 2048],   # hidden dims of decoder
        'epochs' : 600,
        'lr': 1e-4,
        'workspace_dir': '.',
        'encoded_noise': True,
        'model_type': 'cvae',
        'n_critic': 1,
    }

# ----- set parameter
config['dataset'] = 'AWA2'
config['model_type'] = 'cvae'

if config['model_type'] == 'vaegan':
        best_model_path = f"vaegan_pt/{config['dataset']}/best_model_vae.pt"
        last_model_path = f"vaegan_pt/{config['dataset']}/last_model_vae.pt"
elif config['model_type'] == 'cvae':
    best_model_path = f"vaegan_pt/{config['dataset']}/best_model_cvae.pt"
    last_model_path = f"vaegan_pt/{config['dataset']}/last_model_cvae.pt"


# set path
# global_path = "/home/csie2020/p76091543/Plearning_song/"

global_path = "/home/p76091543/Plearning_song/"
resnet101_path = "resnet_direct_2048/"
npy_path = global_path + f"mat_and_model/{config['dataset']}/npy_file/" + resnet101_path

model_mat_path = global_path + f"mat_and_model/{config['dataset']}/" + "two_phase/mat/res_direct_2048.mat"
attr_mat_path = global_path + f"mat_and_model/{config['dataset']}/" + "two_phase/mat/res_attr_direct_2048.mat"

if config['dataset'] == 'SUN':
    config['attr_dim'] = 102
    config['class_num'] = 717
    config['seen_class_num'] = 645
    config['unseen_class_num'] = 72
elif config['dataset'] == 'CUB':
    config['attr_dim'] = 312
    config['class_num'] = 200
    config['seen_class_num'] = 150
    config['unseen_class_num'] = 50
elif config['dataset'] == 'AWA2':
    config['attr_dim'] = 85
    config['class_num'] = 50
    config['seen_class_num'] = 40
    config['unseen_class_num'] = 10


data_train = np.load(npy_path + 'train/train_feature_ft.npy')
attr_train = np.load(npy_path + 'train/train_attr_cms.npy')
label_train = np.load(npy_path + 'train/train_label_list.npy')
print('train')
print(data_train.shape)
print(attr_train.shape)
print(label_train.shape)

data_val = np.load(npy_path + 'val/val_feature_ft.npy')
attr_val = np.load(npy_path + 'val/val_attr_cms.npy')
label_val = np.load(npy_path + 'val/val_label_list.npy')
print('val')
print(data_val.shape)
print(attr_val.shape)
print(label_val.shape)


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


x = torch.from_numpy(data_train)
y = torch.from_numpy(attr_train)
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
# trainer.train()

# model inference
# G_path = "checkpoints/2022-06-10_11-05-12_cvae/D_14.pth"
# trainer.inference(G_path)





# trainer = model.TrainerGAN(config)
# G_path = "checkpoints/2022-06-10_11-05-12_cvae/D_14.pth"
# model = model.TrainerGAN.inference





