import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def Normalization(component):
    nor_temp = (component - component.min()) / (component.max() - component.min())
    return nor_temp

class model(nn.Module):    
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30, 120, device = device),
            nn.ReLU(),
            nn.Linear(120, 1, device = device),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out

    def backward(self, x, y):
        optimizer = optim.Adam(predict.parameters(), lr = 0.001)
        criterion = nn.MSELoss()
        
        loss_cnt = 10000
        
        for epoch in range(loss_cnt):
            predict.train()
            optimizer.zero_grad()
            train_output = predict(x_train)
            train_loss = criterion(train_output.squeeze(), y_train)
            
            if epoch % 100 == 0:
                print('Train loss at {} is {}'.format(epoch, train_loss.item()))
            train_loss.backward()
            optimizer.step()
            
warnings.filterwarnings('ignore')
writer = SummaryWriter()

dtype = torch.float32

cancer = load_breast_cancer()
cancer_data = cancer.data
cancer_label = cancer.target

cancer_feature = cancer.feature_names

cancer_df = pd.DataFrame(data = cancer_data, columns = cancer_feature)
#cancer_df['label'] = cancer_label

cancer_normalization = np.zeros((cancer_df.shape[0], cancer_df.shape[1]))

for i in range(cancer_df.shape[1]):
    cancer_normalization[:, i] = Normalization(cancer_df.iloc[:, i])

x_train, x_test, y_train, y_test = train_test_split(cancer_normalization, cancer_label, test_size = 0.2, random_state = 0)
x_train = torch.FloatTensor(x_train).to(device)
x_test = torch.FloatTensor(x_test).to(device)
y_train = torch.FloatTensor(y_train).to(device)        
            
predict = model().to(device)
predict.backward(x_train, y_train)

writer.close()

with torch.no_grad():
    predict_cancer = predict(x_test)
    predictions = torch.round(torch.sigmoid(predict_cancer)).to(device, dtype=torch.int32)
    y_test = torch.from_numpy(y_test.reshape(-1, 1)).to(device, dtype=torch.int32)
# predictions = predictions.to(y_test.dtype)
print(predictions)
correct_predictions = (predictions == y_test.view(-1, 1)).sum().item()
total_samples = len(y_test)
accuracy = correct_predictions / total_samples
print(f'Accuracy: {accuracy * 100:.2f}%')
            
