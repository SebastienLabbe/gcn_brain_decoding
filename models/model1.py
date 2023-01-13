import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import numpy as np


class GCN(torch.nn.Module):
    def __init__(self, edge_index, edge_weight, n_roi, n_timepoints=1, n_classes=9):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_roi = n_roi
        self.last_out_channels = 4

        #self.batch_norm = m = nn.BatchNorm2d(self.batch_size)

        self.conv1 = tg.nn.ChebConv(
            in_channels=n_timepoints, out_channels=32, K=4, bias=True
        )
        self.conv2 = tg.nn.ChebConv(
            in_channels=32, out_channels=32, K=4, bias=True)
        self.conv3 = tg.nn.ChebConv(
            in_channels=32, out_channels=32, K=4, bias=True)
        self.conv4 = tg.nn.ChebConv(
            in_channels=32, out_channels=32, K=4, bias=True)
        self.conv5 = tg.nn.ChebConv(
            in_channels=32, out_channels=self.last_out_channels, K=4, bias=True)

        self.fc1 = nn.Linear(self.n_roi * self.last_out_channels, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        #print("Begining of loop")
        # print(x.shape)
        #shape = list(x.shape)
        #x = self.batch_norm(x.view(1,*shape)).view(*shape)
        x = self.conv1(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv5(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        # print(x.shape)
        #batch_vector = torch.arange(x.size(0), dtype=int)
        #x = torch.flatten(x, 1)
        # print(x)
        #x = tg.nn.global_mean_pool(x, batch_vector)
        # print(x)
        x = x.view(-1, self.n_roi * self.last_out_channels)
        # print(x.shape)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        # print(x.shape)
        return x
