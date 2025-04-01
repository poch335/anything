import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import warnings

from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

warnings.filterwarnings('ignore')
writer = SummaryWriter()

dtype = torch.float32

train_file = pd.read_csv('C:\code\ML\\train.csv')
test_file = pd.read_csv('C:\code\ML\\test.csv')

def Normalization(component):
    nor_temp = (component - component.min()) / (component.max() - component.min())
    return nor_temp

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.linear_stack = nn.Sequential(
        nn.Linear(6, 16, device = device, dtype = dtype),
        nn.ReLU(),
        nn.Linear(16, 64, device = device, dtype = dtype),
        nn.ReLU(),
        nn.Linear(64, 1, device = device, dtype = dtype),
        nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.linear_stack(x)
        
    def backward(self, x, y):
        optimizer = optim.Adam(predict.parameters(), lr = 0.005)
        loss_fn = nn.MSELoss()
        
        loss_cnt = 5000
        
        for epoch in range(loss_cnt+1):
            optimizer.zero_grad()
            train_output = predict(x_train)
            train_loss = loss_fn(train_output.squeeze(), y_train)
            
            if epoch % 1000 == 0:
                print('Train loss at {} is {}'.format(epoch, train_loss.item()))
            train_loss.backward()
            optimizer.step()
        
### 원하는 데이터열 추출 ###
train = train_file[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
temp = test_file[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare']]

### 문자형 데이터 수치로 변환 ###
gender = {'male' : 0, 'female': 1}
train['Sex'] = train['Sex'].map(gender)
temp['Sex'] = temp['Sex'].map(gender)

### 결측치 제거 ###
train.dropna(inplace = True)
temp.dropna(inplace = True)

train_survived = train.pop('Survived')
train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
test = temp[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

train_nor = Normalization(train)
test_nor = Normalization(test)

x_train = torch.FloatTensor(train_nor.values).to(device)
x_test = torch.FloatTensor(test_nor.values).to(device)
y_train = torch.FloatTensor(train_survived.values).to(device)        

predict = model()
predict.to(device)

predict.backward(x_train, y_train)
writer.close()

with torch.no_grad():
    predict_survived = predict(x_test)
    predictions = torch.round(torch.sigmoid(predict_survived)).to(device, dtype = torch.int32)
print(predictions)
prediction = predictions.cpu().numpy()

