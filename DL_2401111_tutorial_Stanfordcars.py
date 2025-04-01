import os
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import math

import torch.nn.functional as F
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

IMG_SIZE = 64
BATCH_SIZE = 128

# 데이터 경로 설정
data_train_fold = "C:/code/ML/DM/stanfordcars/cars_train"  # cars_train 디렉토리
data_test_fold = "C:/code/ML/DM/stanfordcars/cars_test"

# ImageFolder로 데이터셋 로드
data_train = ImageFolder(root=data_train_fold, transform=transform)
data_test = ImageFolder(root=data_test_fold, transform=transform)

# DataLoader 생성
dataloader = DataLoader(data_train, batch_size=32, shuffle=True)

# 샘플 이미지 시각화 함수
def show_images(dataset, num_samples=20, cols=4):
    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
        img, label = dataset[i]
        img = img.permute(1, 2, 0)  # Tensor 형식 변환 (C, H, W -> H, W, C)
        
        plt.subplot(num_samples // cols + 1, cols, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.show()

# 샘플 이미지 시각화
show_images(data_train)

def linear_beta_schedule(timesteps, start = 0.001, end = 0.02):
    return torch.linspace(start, end, timesteps)
    
def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1, ) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device = "cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]
    
    data_transform = transforms.Compose(data_transforms)
    
    train = data_train
    test = data_test
    
    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image):
    reverse_transforms = transforms.Compoose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.unit8)),
        transforms.ToPILImage()
    ])
    
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))
    
data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)
    


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

image = next(iter(dataloader))[0]

plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward_diffusion_sample(image, t)
    show_tensor_image(img)