# ==============================================================================
#  SEA-ViT Utility Functions
# ==============================================================================

# ==============================================================================
#  SEA-ViT: Sea Surface Current Forecasting with Vision Transformers and BiGRU
#  ----------------------------------------------------------------------------
#  Author   : Teerapong Panboonyuen (Kao)
#  Paper    : "SEA-ViT: Forecasting Sea Surface Currents Using a Vision Transformer
#              and GRU-Based Spatio-Temporal Covariance Model"
#  Version  : 1.0
#  License  : MIT
# ==============================================================================

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SeaCurrentDataset(Dataset):
    def __init__(self, data_npz):
        data = np.load(data_npz)
        self.inputs = data["inputs"]     # [N, T, 2, H, W]
        self.targets = data["targets"]   # [N, forecast_steps, 2]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

def load_dataset(path='data/train.npz', batch_size=8, train=True):
    dataset = SeaCurrentDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

def load_single_sequence(path):
    data = np.load(path)
    x = torch.tensor(data["sequence"]).unsqueeze(0).float()  # [1, T, 2, H, W]
    return x

def save_checkpoint(model, path="checkpoints/sea-vit.pth"):
    torch.save(model.state_dict(), path)

def save_forecast_output(prediction, path):
    np.save(path, prediction.squeeze(0).cpu().numpy())