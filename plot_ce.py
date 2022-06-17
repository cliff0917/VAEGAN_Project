

from unicodedata import decimal
import numpy as np
import os
import scipy.io as sio
from sklearn.metrics import classification_report,confusion_matrix
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='ESZSL')
parser.add_argument('--dataset', type=str, default='SUN')
parser.add_argument('--dataset_path', type=str, default='tf_mat/')
args = parser.parse_args()

res101 = sio.loadmat(args.dataset_path+args.dataset + '/resnet.mat')
attr = sio.loadmat(args.dataset_path+args.dataset + '/attr.mat')

att_splits = attr


if args.dataset == 'AWA2':
    x = np.arange(40)
elif args.dataset == 'CUB':
    x = np.arange(150)
else:
    x = np.arange(645)

# trainval_loc = 'trainval_loc'
# train_loc = 'train_loc'
# val_loc = 'val_loc'
# test_loc = 'test_unseen_loc'

trainval_loc = 'test_seen_loc'
train_loc = 'trainval_loc'
test_loc = 'test_unseen_loc'
labels = res101['labels']



labels = res101['labels']

labels_train = labels[np.squeeze(att_splits[train_loc]-1)]
labels_trainval = labels[np.squeeze(att_splits[trainval_loc]-1)]
labels_test = labels[np.squeeze(att_splits[test_loc]-1)]
all_labels = np.concatenate((labels_trainval, labels_test), axis=0)

train_labels_seen = np.unique(labels_train)
trainval_labels_seen = np.unique(labels_trainval)
test_labels_unseen = np.unique(labels_test)

print("Number of overlapping classes between train and val:", len(
    set(train_labels_seen).intersection(set(test_labels_unseen))))
print("Number of overlapping classes between trainval and test:", len(
    set(trainval_labels_seen).intersection(set(test_labels_unseen))))


i = 0
for labels in train_labels_seen:
    labels_train[labels_train == labels] = i
    i = i+1
k = 0
for labels in trainval_labels_seen:
    labels_trainval[labels_trainval == labels] = k
    k = k+1
l = 0
for labels in test_labels_unseen:
    labels_test[labels_test == labels] = l
    l = l+1


X_features = res101['features']
train_vec = X_features[:, np.squeeze(att_splits[train_loc]-1)]
trainval_vec = X_features[:, np.squeeze(att_splits[trainval_loc]-1)]
test_vec = X_features[:, np.squeeze(att_splits[test_loc]-1)]

print(train_vec.shape)
print(np.mean(train_vec, axis=0))
print(np.std(train_vec, axis=0))

train_mean = np.mean(train_vec, axis=0)
train_std = np.std(train_vec, axis=0)

print(train_labels_seen.shape)
print(trainval_labels_seen.shape)


signature = att_splits['att']
train_sig = signature[:, (train_labels_seen)-1]  # 85 * 40
trainval_sig = signature[:, (trainval_labels_seen)-1]  # 85 * 40
test_sig = signature[:, (test_labels_unseen)-1]  # 85 * 10

print(train_sig.shape)

train_ce_mean = np.mean(train_sig, axis=0)
train_ce_std = np.std(train_sig, axis=0)

print('ce_mean:', np.mean(train_ce_mean))
print('std_mean:', np.mean(train_ce_std))

fig, ax = plt.subplots()

# width = 0.1
# rects1 = ax.plot(x - width/2, train_ce_mean, width, label='std')
# rects2 = ax.plot(x + width/2, train_ce_std, width, label='mean')

rects1 = ax.plot(x, train_ce_mean, 'o-', label='mean')
rects2 = ax.plot(x, train_ce_std, 'o-', label='std')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)

# labels = [str(i) for i in np.arange(10)]
ax.set_ylabel('value')
ax.set_title('ce mean and std')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
fig.savefig(args.dataset_path + args.dataset)