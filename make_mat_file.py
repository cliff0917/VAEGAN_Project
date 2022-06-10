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
args = parser.parse_args()

# data config
dataset = args.dataset
model_type = args.model_type # 'cvae' or 'vae'

# global_path = "/home/csie2020/p76091543/Plearning_song/"
# this_path = "/home/csie2020/p76091543/VAE_Project/"
global_path = "/home/p76091543/Plearning_song/"
this_path = "/home/p76091543/VAEGAN_Project/"
dataset_path = "/SSD/song/"

model_mat_path = this_path + f'vaegan_mat/{dataset}/resnet.mat'
attr_mat_path = this_path + f'vaegan_mat/{dataset}/attr.mat'
print('model_mat_path:', model_mat_path)
print('attr_mat_path:', attr_mat_path)



# classname = pd.read_csv(
#     f'{global_path}/data/{dataset}/classes.txt', header=None, sep='\t')
classname = pd.read_csv(
    dataset_path + f'{dataset}/classes.txt', header=None, sep='\t')

if dataset == 'SUN':
    class_attr_shape = (102, )
    class_attr_dim = 102
    total_class_num = 717
    seen_class_num = 645
    unseen_class_num = 72
elif dataset == 'CUB':
    class_attr_shape = (312, )
    class_attr_dim = 312
    total_class_num = 200
    seen_class_num = 150
    unseen_class_num = 50
elif dataset == 'AWA2':
    class_attr_shape = (85, )
    class_attr_dim = 85
    total_class_num = 50
    seen_class_num = 40
    unseen_class_num = 10
elif dataset == 'plant':
    class_attr_shape = (46, )
    class_attr_dim = 46
    total_class_num = 38
    seen_class_num = 25
    unseen_class_num = 13

resnet101_path = 'resnet_direct_2048/'

npy_path = global_path + f"mat_and_model/{dataset}/npy_file/" + resnet101_path
# model_mat_path = global_path + f"mat_and_model/{dataset}/" + "two_phase/mat/res_direct_2048.mat"
# attr_mat_path = global_path + f"mat_and_model/{dataset}/" + "two_phase/mat/res_attr_direct_2048.mat"


# train_dir = f'{global_path}/data/{dataset}/IMG_backoff/train'
# val_dir = f'{global_path}/data/{dataset}/IMG_backoff/val'
# test_dir = f'{global_path}/data/{dataset}/IMG_backoff/test'
train_dir = f'{dataset_path}/{dataset}/IMG_backoff/train'
val_dir = f'{dataset_path}/{dataset}/IMG_backoff/val'
test_dir = f'{dataset_path}/{dataset}/IMG_backoff/test'
image_size = 224


# Load npy files
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

data_test = np.load(npy_path + 'test/test_feature_ft.npy')
attr_test = np.load(npy_path + 'test/test_attr_cms.npy')
label_test = np.load(npy_path + 'test/test_label_list.npy')




class CustomTensorDataset(TensorDataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __init__(self):
        return len(self.data)



print('#############################################')
# make class
seen_class = next(os.walk(train_dir))[1]
unseen_class = next(os.walk(test_dir))[1]

seen_class.sort()
unseen_class.sort()

print(seen_class)
print(unseen_class)


def make_average_attr(data, label, class_num, predict_attr):
    """
    data: visual feature
    label: label
    class_num: number of classes(e.g. seen num or unseen num)
    predict_attr: semantic embedding
    """
    # predict_attr = model(data)
    sum_attr = [[] for i in range(class_num)]
    count_class = [0 for i in range(class_num)]
    # print('class_name: ', class_num)

    for idx in range(len(predict_attr)):
        l = label[idx]
        if len(sum_attr[l]) == 0:
            sum_attr[l] = predict_attr[idx]
        else:
            sum_attr[l] += predict_attr[idx]
        
        count_class[l] += 1
    
    for i in range(class_num):
        sum_attr[i] = sum_attr[i] / count_class[i]
    
    return np.array(sum_attr)



def assign_attr(seen_class, unseen_class, seen_attr, unseen_attr, total_class_num):
    """
    seen_class: seen classes
    unseen_class: unseen classes
    seen_attr: new average seen attribute
    unseen_attr: new average unsen attribute
    total_class_num: number of total classes(e.g. seen + unseen)
    """
    all_attr = [[] for _ in range(total_class_num)]

    seen_idx = 0
    for v, k in enumerate(seen_class):
        idx = np.where(classname[1] == k)
        all_attr[idx[0][0]] = seen_attr[seen_idx]
        seen_idx += 1
    
    unseen_idx = 0
    for v, k in enumerate(unseen_class):
        idx = np.where(classname[1] == k)
        all_attr[idx[0][0]] = unseen_attr[unseen_idx]
        unseen_idx += 1
    
    return np.array(all_attr)


def convert_dict(seen_class, unseen_class):
    seen_convert_dict = {}
    for v, k in enumerate(seen_class):
        idx = np.where(classname[1] == k)
        seen_convert_dict[v] = idx[0][0]
    
    unseen_convert_dict = {}
    for v, k in enumerate(unseen_class):
        idx = np.where(classname[1] == k)
        unseen_convert_dict[v] = idx[0][0]
        
    return seen_convert_dict, unseen_convert_dict

def correct_label(label, convert_dict):
    for idx in range(len(label)):
        l = label[idx]
        label[idx] = convert_dict[l]
    
    return label


trainer = model.TrainerGAN(config)
E_path = "checkpoints/2022-06-10_14-16-32_cvae/E_39.pth"
model = trainer.inference

# model = torch.load(f"vae_pt/{dataset}/best_model.pt")
# elif model_type == 'cvae':
#     model = torch.load('vae_pt/best_model_cvae.pt')

seen_data = torch.from_numpy(data_train).to(device)
unseen_data = torch.from_numpy(data_test).to(device)


if model_type == 'vae':
    _, seen_mu, _ = model(seen_data)
    _, unseen_mu, _ = model(unseen_data)
elif model_type == 'cvae':
    attr_train_tmp = attr_train
    attr_test_tmp = attr_test
    # attr_train_tmp.astype(float)
    # attr_test_tmp.astype(float)
    origin_seen_attr = torch.from_numpy(attr_train_tmp).float().to(device)
    origin_unseen_attr = torch.from_numpy(attr_test_tmp).float().to(device)
    # _, seen_mu, _ = model(E_path, seen_data, origin_seen_attr)
    # _, unseen_mu, _ = model(E_path, unseen_data, origin_unseen_attr)
    seen_mu = model(E_path, seen_data, origin_seen_attr)
    unseen_mu = model(E_path, unseen_data, origin_unseen_attr)


seen_mu = seen_mu.cpu().detach().numpy()
seen_attr = make_average_attr(data_train, label_train, seen_class_num, seen_mu)

unseen_mu = unseen_mu.cpu().detach().numpy()
unseen_attr = make_average_attr(data_test, label_test, unseen_class_num, unseen_mu)

all_attr = assign_attr(seen_class, unseen_class, seen_attr, unseen_attr, total_class_num)
seen_convert_dict, unseen_convert_dict = convert_dict(seen_class, unseen_class)


converted_label_train = correct_label(label_train, seen_convert_dict)
converted_label_val = correct_label(label_val, seen_convert_dict)
converted_label_test = correct_label(label_test, unseen_convert_dict)



data = np.concatenate((data_train, data_val, data_test), axis=0)
label = np.concatenate((converted_label_train, converted_label_val, 
                        converted_label_test), axis=0)


def mat_list(data, label):
    data_list = []
    label_list = []
    trainval_loc = []
    test_seen_loc = []
    test_unseen_loc = []

    for i in range(len(data)):
        data_list.append(data[i])
        label_list.append(label[i] + 1)
        if i < len(data_train):
            trainval_loc.append(i+1)
        elif i < len(data_train) + len(data_val) and i >= len(data_train):
            test_seen_loc.append(i+1)
        elif i < len(data_train) + len(data_val) + len(data_test) and i >= len(data_train) + len(data_val):
            test_unseen_loc.append(i+1)
        
    data_list = np.row_stack(data_list)
    label_list = np.row_stack(label_list)
    trainval_loc = np.row_stack(trainval_loc)
    test_seen_loc = np.row_stack(test_seen_loc)
    test_unseen_loc = np.row_stack(test_unseen_loc)

    return data_list, label_list, trainval_loc, test_seen_loc, test_unseen_loc

data_list, label_list, trainval_loc, test_seen_loc, test_unseen_loc = mat_list(data, label)


print("save mat:", model_mat_path)   
sio.savemat(model_mat_path, {
    'features': data_list.transpose(),
    'labels': label_list
})


print("save attr:", attr_mat_path)
sio.savemat(attr_mat_path, {
    'trainval_loc': trainval_loc,
    'test_seen_loc': test_seen_loc,
    'test_unseen_loc': test_unseen_loc,
    'att': all_attr.transpose(),
    'train_loc': test_unseen_loc,
    'val_loc': test_unseen_loc
})

