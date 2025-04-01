import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

n_dim = 2

x_train, y_train = make_blobs(n_samples = 80, n_features = n_dim, centers = [[1,1], [-1, -1], [1, -1], [-1, 1]], shuffle = True, cluster_std = 0.3)
x_test, y_test = make_blobs(n_samples = 20, n_features = n_dim, centers = [[1,1], [-1, -1], [1, -1], [-1, 1]], shuffle = True, cluster_std = 0.3)

print(y_train)
print(y_test)

def label_map(y):
    mapping_dict = {0: 0, 1: 0, 2: 1, 3: 1}
    return np.vectorize(mapping_dict.get)(y)

def vis_data(x, y = None, c = 'r'):
    if y is None:
        y = [None] * len(x)
    for x_, y_ in zip(x, y):
        if y_ is None:
            plt.plot(x_[0], x_[1], '*', markerfacecolor = 'none', markeredgecolor = c)
        else:
            plt.plot(x_[0], x_[1], c+'o' if y_ == 0 else c+'+')
    

y_train = label_map(y_train)
y_test = label_map(y_test)

print(y_train)
print(y_test)

plt.figure()
vis_data(x_train, y_train, c = 'r')
plt.show()

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

class model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(model, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
                
        self.w1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_tensor):
        out = self.relu(self.w1(input_tensor))
        out = self.sigmoid(self.w2(out))
        return out
    
        
predict = model(2, 5)

learning_rate = 0.03
criterion = nn.MSELoss()

epochs = 20000
optimizer = optim.Adam(predict.parameters(), lr = learning_rate)

predict.eval()

test_loss_before = criterion(predict(x_test).squeeze(), y_test)
print('Before Training, test loss is {}'.format(test_loss_before.item()))      

for epoch in range(epochs):
    predict.train()
    optimizer.zero_grad()
    train_output = predict(x_train)
    train_loss = criterion(train_output.squeeze(), y_train)
    
    if epoch % 100 == 0:
        print('Train loss at {} is {}'.format(epoch, train_loss.item()))
        
    train_loss.backward()
    optimizer.step()
    
        
predict.eval()
test_loss = criterion(torch.squeeze(predict(x_test)), y_test)
print('After Training, test loss is {}'.format(test_loss.item()))
