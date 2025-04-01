import torch
import torch.nn as nn
import torch.optim as optim

from sklearn import datasets

dataset = datasets.load_boston()

x, y = dataset['data'], dataset['target']

# 입력 데이터와 타겟을 tensor 자료구조로 변환
x = torch.FloatTensor(x)
y = torch.FloatTensor(y).unsqueeze(-1)

# 표준화
x = (x - torch.mean(x)) / torch.std(x)

model = nn.Sequential(
    nn.Linear(13, 39),
    nn.Linear(39, 1)
    )

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(1, 10001):
    optimizer.zero_grad()
    
    y_predicted = model(x)
    
    loss = criterion(y_predicted, y)
    
    loss.backward()
    
    optimizer.step()
    
    if epoch % 100 == 0:
        print('{}epoch, loss: {:.4f}'.format(epoch, loss.item()))