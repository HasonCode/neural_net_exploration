

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
from tqdm import tqdm
import torchmetrics
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from unipen_class import Unipen, build_index

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels = input_channels, out_channels=32, kernel_size = 5, padding=2)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels=64, kernel_size = 5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels = 64, out_channels=128, kernel_size = 3, padding=1)
        self.fc1 = nn.Linear(64*128, num_classes)#64 * 12

    def forward(self, x):
        """
        Performs computation on the input stroke tensor
        Parameters:
            x: Input Tensor
        
        Returns:
            torch.Tensor
                Output tensor after going through network
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        return x


dataset = Unipen(
        root="unipen_data/unipen/CDROM/train_r01_v07/data",
        index_path="unipen_index.json",
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()-_=+[]{}|,.<>?;:`\\",
        target_mode="char",  # or "sequence" for multi-character sequences
        max_points=512,  # Maximum number of points per sample
        normalize=True  # Normalize coordinates to [0, 1]
    )

dataloader = DataLoader(
    dataset = dataset,
    batch_size = 32,
    shuffle = True,
    num_workers=4
)


print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(3, len("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()-_=+[]{}|,.<>?;:`\\")).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()

data_iter = iter(dataloader)

epochs = 30

for mini_epoch in range(epochs):
    model.train()
    running_loss = 0
    batch_count = 1
    for batch_x, batch_y in dataloader:

        batch_x = batch_x.permute(0,2,1)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        print(f"Batch {batch_count} loss: {loss.item():.8f}")
        batch_count+=1
        running_loss += loss.item() * batch_x.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {mini_epoch + 1} loss: {epoch_loss:.8f}")

torch.save(model.state_dict(),"models/cnn_unipen_handwriting")

