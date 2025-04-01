import torch
import torch.nn as nn
import torch.optim as optim

import os 
import glob

import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import random_split, TensorDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
def calculate_pixels(image_path):
    with Image.open(image_path) as img:
        return img.size[0] * img.size[1] 
    
img_path = Path("img")
img_path_list = list(img_path.glob("*/*.jpg"))
classes = sorted(os.listdir(img_path))

print(f'Total Classes = {len(classes)}')

print(f'Total Image = {len(img_path_list)}')

for label in classes:
    img_per_class = list(Path(os.path.join(img_path, label)).glob('*.jpg'))
    print(f'* {label}: {len(img_per_class)} images')
    
labels = [img_path.parent.stem for img_path in img_path_list]
data = pd.DataFrame({'Image': img_path_list,
                     'Target': labels})

SEED = 42

dataset = ImageFolder(root = 'img/',
                      transform = transforms.Compose([transforms.ToTensor(),]))

data_train, data_test = train_test_split(dataset, test_size=0.3, random_state = SEED)
train_loader = DataLoader(data_train, batch_size = 64, shuffle = True)
test_loader = DataLoader(data_test, batch_size = 64, shuffle = False)
images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

label2id = dict(zip(classes, range(len(classes))))

class model(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3 * 256 * 256, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 15)
        )
        
    def forward(self, x):
        return self.layer(x)
    
    def backward(images, labels):
        optimizer = optim.Adam(classification_model.parameters(), lr = 0.0001, weight_decay = 0.0001)
        criterion = nn.CrossEntropyLoss()
        # model.train()
        loss_cnt = 5000
        
        
        for epoch in range(loss_cnt+1):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                print(type(images), type(labels))
                labels = torch.tensor(labels)
                images = images.view(images.size(0), -1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if epoch % 100 == 0:
                    print('Train loss at {} is {}'.format(epoch, loss.item()))

 
classification_model = model(3 * 256 * 256).to(device)

classification_model.backward(train_loader)

classification_model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.size(0), -1)
        outputs = classification_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')
