import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


iris_load = load_iris()

iris = pd.DataFrame(data = iris_load.data, columns = iris_load.feature_names)
target = pd.DataFrame(data = iris_load.target)
target = target.iloc[:, 0].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

target_ohe = pd.get_dummies(target)

x_train, x_test, y_train, y_test = train_test_split(iris, target_ohe, test_size = 0.2, random_state = 0)

x_train = torch.FloatTensor(x_train.values).to(device)
x_test = torch.FloatTensor(x_test.values).to(device)
y_train = torch.tensor(y_train.values.argmax(axis=1), dtype = torch.long).to(device)
y_test = torch.tensor(y_test.values.argmax(axis=1), dtype = torch.long).to(device)



class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
            )
        
    def forward(self, x):
        return self.linear_stack(x)
    
    def backward(self, x, y):
        optimizer = optim.Adam(classification.parameters(), lr = 0.01)
        criterion = nn.CrossEntropyLoss()
        
        epochs = 500
        
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            train_output = classification(x)
            train_loss = criterion(train_output, y)
            
            if epoch % 10 == 0:
                print('Train loss at {} is {}'.format(epoch, train_loss.item()))
            
            train_loss.backward()
            optimizer.step()
        
        
classification = model().to(device)
classification.backward(x_train, y_train)

with torch.no_grad():
    correct = 0
    total = 0
    
    classification_predict = classification(x_test)
    _, predicted = torch.max(classification_predict.data, 1)
    total += x_test.size(0)
    correct += (predicted == y_test).sum().item()
    
    accuracy = 100 * correct / total
    print()
    print(f'Accuracy of the model on the test images: {accuracy}%')
    