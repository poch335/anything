import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder

import glob
import numpy as np
import pandas as pd
import warnings

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


warnings.filterwarnings('ignore')

dtype = torch.float32

#torch.optimizer
#torch.trainning

loss_cnt = 0

# train_file = np.loadtxt(f'train.csv', delimiter=',', skiprows=1, usecols = (1, 2, 5, 6, 10), dtype = str)
# test_file = np.loadtxt(f'test.csv', delimiter=',', skiprows=1, dtype = str)


# train = train_file
# test = test_file

# train[:, 2] = np.where(train[:, 2] == 'male', 1, 2)
# train = train.astype(np.float32)


# train = np.ma.masked_invalid(train)
# test = np.ma.masked_invalid(test)

# x_train = torch.FloatTensor(train)
# y_train = torch.FloatTensor(test)





class model(nn.Module, device, dtype):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(5, 128, device = device, dtype = dtype)
        self.w2 = nn.Linear(128, 256, device = device, dtype = dtype)
        self.w3 = nn.Linear(256, 1, device = device, dtype = dtype)
        
    def forward(self, x, y):
        temp_x1 = self.w1(x)
        temp_x2 = self.w2(temp_x1)
        return self.w3(temp_x2)
        
    def backward(self, x, y, loss_fn):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), ir = 0.001)
        
        loss = 1
        
        while loss > 1e-7:
            pred = model(x)
            loss = loss_fn(pred, y)
            
            optimizer.zero_grad()    
            loss.backward() 
            optimizer.step()
            
            global loss_cnt
            loss_cnt += 1
            
            if loss_cnt % 100 == 0:
                print('Epoch {:4d}/{} Cost: {:.6f'.format(loss_cnt, loss.item()))
            
predict = model()