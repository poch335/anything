import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
import matplotlib.pyplot as plt

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드 및 전처리
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
data = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
data.dropna(inplace=True)  # 결측치 제거

# 범주형 데이터 인코딩
for col in data.columns:
    if data[col].dtype == 'object':
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        
# 특성과 레이블 분리
X = data.drop('income', axis=1).values
y = data['income'].values

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# PyTorch 데이터셋 준비
X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)

# 오토인코더 모델 정의
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(X_train.shape[1], 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 25),
            nn.ReLU(),
            nn.Linear(25, 50),
            nn.ReLU(),
            nn.Linear(50, X_train.shape[1]),
            nn.Sigmoid()  # 출력을 [0, 1] 범위로 조정
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 모델 초기화 및 학습
model = Autoencoder().to(device)
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 2000
y_cosine = torch.ones(X_train_torch.size(0)).to(device)

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, X_train_torch, y_cosine)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 모델을 평가 모드로 설정
model.eval()

# SHAP 값 계산 - SHAP은 CPU에서 계산
background = X_train_torch[:100].to('cpu')  # 배경 데이터를 CPU로 이동
e = shap.DeepExplainer(model.to('cpu'), background)  # 모델도 CPU로 이동
shap_values = e.shap_values(X_test_torch[:10].to('cpu'))

# SHAP 요약 플롯
shap.summary_plot(shap_values, X_test[:10], feature_names=columns[:-1])
