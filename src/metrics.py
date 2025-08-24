# ==============================================================================
#  SEA-ViT Evaluation Metrics
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_rmse(preds, targets):
    return np.sqrt(mean_squared_error(targets, preds))

def compute_mae(preds, targets):
    return mean_absolute_error(targets, preds)

def compute_corr(preds, targets):
    return np.corrcoef(preds.flatten(), targets.flatten())[0, 1]