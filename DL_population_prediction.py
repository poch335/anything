import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import warnings

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
warnings.filterwarnings('ignore')
writer = SummaryWriter()

dtype = torch.float32

data_file = pd.read_csv('C:\code\\ML\\population\\world_population.csv')

def Normalization(component):
    nor_temp = (component - component.min()) / (component.max() - component.min())
    return nor_temp

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(device = device),
            nn.Relu(),
            nn.Linear(device = device),
            nn.ReLu(),
            nn.Linear(device = device),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        out = self.linear_stack(x)
        return out
    
    def backward(self, x, y):
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        criterion = nn.MSELoss()
        
        loss_cnt = 50000
        
        for epoch in range(loss_cnt):
            model.train()
            optimizer.zero_grad()
            train_output = model(x_train)
            train_loss = criterion(train.output, y_train)
        
            if epoch % 100 == 0:
                print('Train loss at {} is {}'.format(epoch, train_loss.item()))
            train_loss.backward()
            optimizer.step()

ohe = pd.get_dummies(data_file)

data_normalization = np.zeros((data_file.shape[0], data_file.shape[1]))

for i in range(data_file.shape[1]-1):
    data_normalization[:, i+1] = Normalization(ohe.iloc[:, i+1])
            
train = data_normalization[['country', 'continent', '2022 population', '2020 population', '2015 population', '2010 population', '2000 population', '1990 population', '1980 population', '1970 population', 'area', 'density', 'growth rate']]
test = data_file[['2023 population']]
            
x_train, x_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 1)

print(x_test)

x_train = torch.FloatTensor(x_train.values).to(device)
x_test = torch.FloatTensor(x_test.values).to(device)
y_train = torch.FloatTensor(y_train.values).to(device)

predict = model()
predict.backward(x_train, y_train)

with torch.no_grad():
    predict_population = predict(x_test)
    predictions = torch.round(torch.sigmoid(predict_population))

print(predictions)




