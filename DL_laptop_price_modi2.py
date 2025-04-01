import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import warnings

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

warnings.filterwarnings('ignore')
writer = SummaryWriter()

dtype = torch.float32

data_file = pd.read_csv('C:\\code\\ML\\laptop_price\\Laptop_price.csv')

def Normalization(component):
    nor_temp = (component - component.min()) / (component.max() - component.min())
    return nor_temp

def Standardization(component):
    std_component = (component - np.mean(component)) / np.std(component)
    return std_component

class Model(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dimension, 8, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(8, 1, device=device, dtype=dtype)
        )

    def forward(self, x):
        out = self.linear_stack(x)
        return out

    def train_model(self, x, y, epochs=1000, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CosineEmbeddingLoss()
        
        target = torch.ones(x.size(0)).to(device)  # Assuming all pairs are similar
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            train_output = self(x)
            train_loss = criterion(train_output, y, target)
        
            if epoch % 100 == 0:
                print('Train loss at epoch {} is {}'.format(epoch, train_loss.item()))
            train_loss.backward()
            optimizer.step()

laptop_component = data_file[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']]
laptop_price = data_file[['Price']]

# Normalize the data
for i in range(laptop_component.shape[1]):
    laptop_component.iloc[:, i] = Normalization(laptop_component.iloc[:, i])
laptop_price = Normalization(laptop_price)

x_train, x_test, y_train, y_test = train_test_split(laptop_component, laptop_price, test_size=0.2, random_state=0)
raw_price = y_test.to_numpy()

x_train = torch.FloatTensor(x_train.values).to(device)
x_test = torch.FloatTensor(x_test.values).to(device)
y_train = torch.FloatTensor(y_train.values).to(device, dtype=dtype)

model = Model(laptop_component.shape[1])
model.to(device)
model.train_model(x_train, y_train)

with torch.no_grad():
    predict_price = model(x_test)
    predict_price_np = predict_price.cpu().numpy()
    y_test = torch.from_numpy(y_test.values.reshape(-1, 1)).to(device, dtype=dtype)

print(predict_price_np)
Error_percentage = raw_price / predict_price_np
high_Error_percentage = max(Error_percentage)

print(f'Highest Error Percentage: {high_Error_percentage}')
