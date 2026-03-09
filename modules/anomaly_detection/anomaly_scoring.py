import torch
import numpy as np

def compute_anomaly_scores(model, sequences):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    device = next(model.parameters()).device
    X = torch.tensor(sequences, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(X)
        loss = torch.mean((X - reconstructed) ** 2, dim=(1, 2))

    return loss.cpu().numpy()