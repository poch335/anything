# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:42:23 2024

@author: user
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from torch.utils.data import DataLoader, TensorDataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 시드 고정
seed_everything(0)
    
# 데이터셋 로드
# 'data.csv' 파일을 읽어올 것으로 가정합니다. 파일 경로는 실제 데이터셋의 위치에 맞게 조정해야 합니다.
df = pd.read_csv('4.csv')

df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d_%H:%M:%S.%f')

# 첫 번째 열을 인덱스 (시간)으로 설정
df.set_index('Time', inplace=True)

# 'Time' 열을 제외한 데이터만을 numpy 배열로 변환
data = df.values

# 데이터를 이미지로 표시
# plt.figure(figsize=(12, 6))  # 이미지의 크기를 조정합니다.
# plt.imshow(data, aspect='auto', cmap='viridis')  # aspect='auto'는 축의 스케일을 데이터에 맞춰 조정합니다.
# plt.colorbar()  # 데이터의 값 범위에 대한 컬러 바를 추가합니다.
# plt.xlabel('wavelength')
# plt.ylabel('Time Index')
# plt.title('Intensity Heatmap Over Time')
# plt.show()  
    
# 'Time' 열을 제외하고 numpy 배열로 변환
# data = df.drop('Time', axis=1).values
# 데이터를 PyTorch 텐서로 변환
tensor_data = torch.tensor(data, dtype=torch.float32).to(device)

# 데이터셋 준비
dataset = TensorDataset(tensor_data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(3648, 512, device=device),
            nn.Linear(512, 128, device=device),
            nn.Linear(128, 64, device=device),
            nn.Linear(64, 16, device=device),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 64, device=device),
            nn.Linear(64, 128, device=device),
            nn.Linear(128, 512, device=device),
            nn.Linear(512, 3648, device=device),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# 모델 인스턴스 생성
autoencoder = Autoencoder().to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir = 'logs')
def train_autoencoder(model, data_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for data in data_loader:
            inputs = data[0].to(device)
            # Forward pass
            outputs, _= model(inputs)
            loss = criterion(outputs, inputs, torch.tensor([1],device=device))
            writer.add_scalar('loss', loss.item(),epoch)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
# 훈련 실행
train_autoencoder(autoencoder, data_loader, criterion, optimizer, epochs=1000)

# 테스트 데이터셋을 사용하여 복호화
# 여기서는 전체 데이터셋을 가정합니다.
test_dataset = TensorDataset(tensor_data)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# decoded_batches = []
hidden_states = []

autoencoder.eval()  # 모델을 평가 모드로 설정
with torch.no_grad():
    for inputs in data_loader:    # data_loader는 입력 데이터와 함께 타겟 데이터도 반환합니다.
        # 배치 데이터만 모델에 전달합니다.
        decoded, encoded = autoencoder(inputs[0].to(device))
        hidden_states.append(encoded.cpu())
        # decoded_batch = autoencoder(inputs[0])
        # 복호화된 배치를 리스트에 추가합니다.
        # decoded_batches.append(decoded_batch.cpu())

# 모든 복호화된 배치를 하나의 데이터 세트로 합칩니다.
hidden_states_data = torch.cat(hidden_states, dim = 0)
# decoded_data = torch.cat(decoded_batches, dim=0)

# 이제 decoded_data를 사용하여 시각화를 진행합니다.
plt.figure(figsize=(12, 6))
plt.imshow(hidden_states_data.numpy(), aspect='auto', cmap='gray')
plt.colorbar()
plt.xlabel('Wavelength')
plt.ylabel('Samples')
plt.title('Hidden state')
plt.show()