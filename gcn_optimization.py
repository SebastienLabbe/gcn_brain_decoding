#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from nilearn import plotting
import nilearn.connectome
from nilearn import datasets
from nilearn.input_data import NiftiMasker
import warnings
from gcn_windows_dataset import TimeWindowsDataset
from gcn_model import GCN
from graph_construction import make_group_graph
import sys
import shutil
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch.utils.data import DataLoader
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


wdir = os.getcwd()
sys.path.append(wdir + '/src')


# In[3]:


warnings.filterwarnings(action='once')


wdir = os.getcwd()

# We are fetching the data for subject 4
data_dir = os.path.join(wdir, 'data')
sub_no = 2
haxby_dataset = datasets.fetch_haxby(
    subjects=[sub_no], fetch_stimuli=True, data_dir=data_dir)
func_file = haxby_dataset.func[0]

# Standardizing
mask_vt_file = haxby_dataset.mask_vt[0]
masker = NiftiMasker(mask_img=mask_vt_file, standardize=True)

# cognitive annotations
behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=' ')
X = masker.fit_transform(func_file)
y = behavioral['labels']


# In[4]:


categories = y.unique()
print(categories)
print('y:', y.shape)
print('X:', X.shape)


# In[5]:


# Estimating connectomes and save for pytorch to load
connectome_measure = nilearn.connectome.ConnectivityMeasure(kind="correlation")
conn = connectome_measure.fit_transform([X])[0]
corr_matrix_z = np.tanh(connectome_measure.mean_)  # convert to z-score
sig = 0.25
# a Gaussian kernel, defined in Shen 2010
corr_matrix_z = np.exp(corr_matrix_z / sig)

n_regions_extracted = X.shape[-1]
title = 'Correlation between %d regions' % n_regions_extracted

print('Correlation matrix shape:', conn.shape)

# First plot the matrix
display = plotting.plot_matrix(corr_matrix_z, vmax=1, vmin=-1,
                               colorbar=True, title=title)


# In[6]:


adj_sparse = tg.utils.dense_to_sparse(torch.from_numpy(corr_matrix_z))
graph = tg.data.Data(edge_index=adj_sparse[0], edge_attr=adj_sparse[1])


# # Prepare Data

# In[7]:


# generate data

# cancatenate the same type of trials
concat_bold = {}
for label in categories:
    cur_label_index = y.index[y == label].tolist()
    curr_bold_seg = X[cur_label_index]
    concat_bold[label] = curr_bold_seg


# In[8]:


# split the data by time window size and save to file
window_length = 2
dic_labels = {name: i for i, name in enumerate(categories)}

# set output paths
split_path = os.path.join(data_dir, 'haxby_split_win/')
if not os.path.exists(split_path):
    os.makedirs(split_path)
out_file = os.path.join(split_path, '{}_{:04d}.npy')
out_csv = os.path.join(split_path, 'labels.csv')

label_df = pd.DataFrame(columns=['label', 'filename'])
for label, ts_data in concat_bold.items():
    ts_duration = len(ts_data)
    ts_filename = f"{label}_seg"
    valid_label = dic_labels[label]

    # Split the timeseries
    rem = ts_duration % window_length
    n_splits = int(np.floor(ts_duration / window_length))

    ts_data = ts_data[:(ts_duration - rem), :]

    for j, split_ts in enumerate(np.split(ts_data, n_splits)):
        ts_output_file_name = out_file.format(ts_filename, j)

        split_ts = np.swapaxes(split_ts, 0, 1)
        np.save(ts_output_file_name, split_ts)

        curr_label = {'label': valid_label,
                      'filename': os.path.basename(ts_output_file_name)}
        label_df = label_df.append(curr_label, ignore_index=True)

label_df.to_csv(out_csv, index=False)


# # Data Loaders

# In[9]:


# split dataset#!/usr/bin/env python
# coding: utf-8

# In[1]:


InteractiveShell.ast_node_interactivity = "all"


# In[2]:


wdir = os.getcwd()
sys.path.append(wdir + '/src')

random_seed = 0

train_dataset = TimeWindowsDataset(
    data_dir=split_path,
    partition="train",
    random_seed=random_seed,
    pin_memory=True,
    normalize=True,
    shuffle=True)

valid_dataset = TimeWindowsDataset(
    data_dir=split_path,
    partition="valid",
    random_seed=random_seed,
    pin_memory=True,
    normalize=True,
    shuffle=True)

test_dataset = TimeWindowsDataset(
    data_dir=split_path,
    partition="test",
    random_seed=random_seed,
    pin_memory=True,
    normalize=True,
    shuffle=True)

print("train dataset: {}".format(train_dataset))
print("valid dataset: {}".format(valid_dataset))
print("test dataset: {}".format(test_dataset))


batch_size = 10

torch.manual_seed(random_seed)
train_generator = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
valid_generator = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True)
test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_features, train_labels = next(iter(train_generator))
print(
    f"Feature batch shape: {train_features.size()}; mean {torch.mean(train_features)}")
print(
    f"Labels batch shape: {train_labels.size()}; mean {torch.mean(torch.Tensor.float(train_labels))}")


# # Define Model

# In[11]:


# In[12]:


gcn = GCN(graph.edge_index,
          graph.edge_attr,
          n_roi=X.shape[1],
          n_timepoints=window_length,
          n_classes=len(categories))


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * dataloader.batch_size

        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= X.shape[0]
        if (batch % 10 == 0) or (current == size):
            print(
                f"#{batch:>5};\ttrain_loss: {loss:>0.3f};\ttrain_accuracy:{(100*correct):>5.1f}%\t\t[{current:>5d}/{size:>5d}]")


def valid_test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model.forward(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= size
    correct /= size

    return loss, correct


avg_acc_train = []
avg_acc_test = []
loss_trains = []
loss_tests = []


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    gcn.parameters(), lr=0.00003, weight_decay=5e-4, amsgrad=True)


epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}\n-------------------------------")
    train_loop(train_generator, gcn, loss_fn, optimizer)
    loss_train, correct_train = valid_test_loop(valid_generator, gcn, loss_fn)
    loss_test, correct_test = valid_test_loop(test_generator, gcn, loss_fn)
    avg_acc_train.append(correct_train)
    avg_acc_test.append(correct_test)
    loss_trains.append(loss_train)
    loss_tests.append(loss_test)
    print(
        f"Valid train metrics:\n\t avg_loss: {loss_train:>8f};\t avg_accuracy: {(100*correct_train):>0.1f}%")
    print(
        f"Valid test metrics:\n\t avg_loss: {loss_test:>8f};\t avg_accuracy: {(100*correct_test):>0.1f}%")

data = {
    "avg_acc_train": avg_acc_train,
    "avg_acc_test": avg_acc_test,
    "loss_trains": loss_trains,
    "loss_tests": loss_tests,
    "model": gcn,
    "loss_fn": loss_fn,
    "optimizer": optimizer,
    "epochs": epochs,
}


pickle.dump(data,)
