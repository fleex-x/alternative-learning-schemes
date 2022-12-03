import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np

class TwoLayersModel(nn.Module):
    def __init__(self, num_classes, in_features, hidden_layer=10) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_layer), 
            nn.ReLU()
        )
        self.layer2 = nn.Linear(in_features=hidden_layer, out_features=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))

class CustomDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.int32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array([self.y[idx]])) 