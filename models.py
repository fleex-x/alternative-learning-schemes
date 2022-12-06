import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np

class NNModel(nn.Module):
    def __init__(self, num_classes, in_features, hidden_layers = [20, 30]) -> None:
        super().__init__()
        lst = in_features
        self.model: nn.Sequential = nn.Sequential()
        for neurons in hidden_layers:
            self.model.append(nn.Linear(in_features=lst, out_features=neurons))
            self.model.append(nn.ReLU())
            lst = neurons
        self.model.append(nn.Linear(in_features=lst, out_features=num_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class CustomDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.int32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array([self.y[idx]])) 


class LeNet(nn.Module):
    
    def __init__(self, in_features=3, num_classes=10):
        super(LeNet, self).__init__()
        self.conv_block = nn.Sequential( 
            nn.Conv2d(in_channels=in_features, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        
        self.linear_block = nn.Sequential( 
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_block(x)
        return x