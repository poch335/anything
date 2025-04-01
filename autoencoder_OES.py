import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd
import os 
import glob

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dtype = torch.float32

data_path = 'Data'

file_list = glob.glob(f'{data_path}/*.csv')
raw_data = np.loadtxt(file_list, delimiter = ',')


class Autoencoder(nn.Module):
    def __init__(self, x_size, y_size):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(3648 * y_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3648 * y_size)
            )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def backward(autoencoder, train_loader, y_size):
        autoencoder.train()
        for step, (x, label) in enumerate(train_loader):
            # x = x.view(-1, 3648 * ).to(device)
            # y = x.view(-1, 3648 * ).to(device)
            label = label.to(device)
            
            encoded, decoded = autoencoder(x)
            
            loss = criterion(decoded, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

x_train, x_test, y_train, y_test = train_test_split(raw_data.loc[:, 1:], test_size = 0.2, random_state = 0)
            
data_x_size, data_y_size = raw_data.shape[0], raw_data.shape[1]



result_encoder = Autoencoder(data_x_size, data_y_size)
result_encoder.backward(data_y_size)
 