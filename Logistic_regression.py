import torch
import torch.nn as nn
import torch.optim as optim

from sklearn import datasets

dataset = datasets.load_breast_cancer()

x, y = dataset['data'], dataset['target']

# 입력 데이터와 타겟을 tensor 자료구조로 변환
x = torch.FloatTensor(x)
y = torch.FloatTensor(y).view(-1, 1)

# 표준화
x = (x - torch.mean(x)) / torch.std(x)

model = nn.Sequential(
    nn.Linear(30, 1),
    nn.Sigmoid()
    )

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

for epoch in range(1, 10001):
    optimizer.zero_grad()
    
    y_predicted = model(x)
    
    loss = criterion(y_predicted, y)
    
    loss.backward()
    
    optimizer.step()
    
    if epoch % 100 == 0:
        print('{}epoch, loss: {:.4f}'.format(epoch, loss.item()))
        
y_predicted = (model(x) >= 0.5).float()

score = (y_predicted == y).float().mean()
print(score)