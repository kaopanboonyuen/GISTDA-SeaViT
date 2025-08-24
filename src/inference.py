# ==============================================================================
#  SEA-ViT Inference Script
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
from src.model import SEAViT
from src.utils import load_single_sequence, save_forecast_output

def infer(model_path, input_path, output_path):
    model = SEAViT()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = load_single_sequence(input_path)  # [1, T, C, H, W]
    with torch.no_grad():
        prediction = model(x)  # [1, forecast_steps, 2]
    
    save_forecast_output(prediction, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    infer(args.model_path, args.input_path, args.output_path)