from ast import walk
import numpy as np
import pandas as pd
import torch
import random
import scipy.io as sio
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import argparse
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
# torch.cuda.set_device('cuda:1')

parser = argparse.ArgumentParser(description="make_mat")
parser.add_argument("--model_type", type=str, default="cvae")
parser.add_argument("--dataset", type=str, default="AWA2")
parser.add_argument("--E_path", type=str, default="methods/method_cvae/checkpoints/AWA2/2022-06-23_02-14-41_cvae/E.pth")
args = parser.parse_args()


# data config
dataset = args.dataset
model_type = args.model_type # 'cvae' or 'vae'

config['dataset'] = dataset
config['model_type'] = model_type

# global_path = "/home/csie2020/p76091543/Plearning_song/"
# this_path = "/home/csie2020/p76091543/VAE_Project/"
home_dir = os.path.expanduser("~")

# global_path is plearning path
global_path = os.path.join(home_dir, 'Plearning_song/')
gan_path = os.path.join(home_dir, "VAEGAN_Project/")

if home_dir == "/home/csie2020/p76091543":
    dataset_path = os.path.join(global_path, "data/")
else:
    dataset_path = "/SSD/song/"

dataset_path = os.path.join(global_path, "data/")

model_mat_path = gan_path + f'mat/{dataset}/resnet.mat'
attr_mat_path = gan_path + f'mat/{dataset}/attr.mat'
print('model_mat_path:', model_mat_path)
print('attr_mat_path:', attr_mat_path)



# classname = pd.read_csv(
#     f'{global_path}/data/{dataset}/classes.txt', header=None, sep='\t')
classname = pd.read_csv(
    dataset_path + f'/{dataset}/classes.txt', header=None, sep='\t')

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



trainer = model.TrainerGAN(config)
# E_path = gan_path + "checkpoints/2022-06-12_05-07-18_cvae/E_249.pth"
E_path = gan_path + args.E_path
model = trainer.inference

# model = torch.load(f"vae_pt/{dataset}/best_model.pt")
# elif model_type == 'cvae':
#     model = torch.load('vae_pt/best_model_cvae.pt')

path = f'other_mats/{dataset}/attr.mat'

mat_res = sio.loadmat(f'other_mats/{dataset}/resnet.mat')
feature = mat_res['features'].T
label = mat_res['labels'].astype(int).squeeze() - 1

mat_attr = sio.loadmat(path)
trainval_loc = mat_attr['trainval_loc'].squeeze() - 1
train_loc = mat_attr['train_loc'].squeeze() - 1
val_unseen_loc = mat_attr['val_loc'].squeeze() - 1
test_seen_loc = mat_attr['test_seen_loc'].squeeze() - 1
test_unseen_loc = mat_attr['test_unseen_loc'].squeeze() - 1

class_num = config['class_num']

# Get attr
attribute = torch.from_numpy(mat_attr['att'].T).float().to(device)
attribute = attribute[label]

features = mat_res['features'].transpose()
# Predict
features = torch.from_numpy(features).float().to(device)

assert features.size(0) == attribute.size(0)

predict_attr = model(E_path, features, attribute)
predict_attr = predict_attr.cpu().detach().numpy()
labels = mat_res['labels']

sum_attr = [[] for i in range(class_num)]
real_attr = [[] for i in range(class_num)]
count_class = [0 for i in range(class_num)]
for idx in range(len(predict_attr)):
    l = int(labels[idx]) - 1
    if sum_attr[l] == []:
        sum_attr[l] = predict_attr[idx].copy()
    else:    
        sum_attr[l] += predict_attr[idx]
    count_class[l] += 1

# averge
for i in range(class_num):
    sum_attr[i] = sum_attr[i] / count_class[i]
sum_attr = np.array(sum_attr)


print(sum_attr.shape)


mat_attr['att'] = sum_attr.transpose()
sio.savemat(attr_mat_path, mat_attr)
