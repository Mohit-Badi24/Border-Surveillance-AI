import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from modules.anomaly_detection.lstm_autoencoder import LSTMAutoencoder

def train_model(sequences, epochs=30, batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(sequences, dtype=torch.float32)

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMAutoencoder(input_dim=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for batch in loader:
            batch = batch[0].to(device)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "models/anomaly_lstm.pt")
    print("Model saved as anomaly_lstm.pt")

    return model