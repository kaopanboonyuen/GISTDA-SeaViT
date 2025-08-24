# ==============================================================================
#  SEA-ViT: Sea Surface Current Forecasting with Vision Transformers and BiGRU
#  ----------------------------------------------------------------------------
#  Author   : Teerapong Panboonyuen (Kao)
#  Paper    : "SEA-ViT: Forecasting Sea Surface Currents Using a Vision Transformer
#              and GRU-Based Spatio-Temporal Covariance Model"
#  Version  : 1.0
#  License  : MIT
# ==============================================================================

# Train SEA-ViT
from src.model import SEAViT
from src.utils import load_dataset, save_checkpoint
from src.metrics import compute_rmse

def train():
    # Setup
    model = SEAViT()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    train_loader = load_dataset(train=True)
    
    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        save_checkpoint(model)

if __name__ == "__main__":
    train()