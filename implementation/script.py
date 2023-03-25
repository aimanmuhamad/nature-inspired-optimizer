import torch
import torch.nn as nn
import numpy as np
from mlp import MLP
from optimizer.aco import AntColonyOptimization as ACO
from variables import n_input_dim, n_hidden1, n_hidden2, n_output

X_train = torch.tensor([[0, 0, 1, 0], [1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 1, 1]], dtype=torch.float32)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

model = MLP(n_input_dim, n_hidden1, n_hidden2, n_output)
optimizer = ACO(model.parameters(), lr=0.1, ant_count=10, evaporation_rate=0.5, Q=10, beta=1, elite_ant_count=1)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = nn.BCELoss()(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    # Evaluasi performa pada setiap epoch
    with torch.no_grad():
        y_pred = model(X_train)
        y_pred = (y_pred > 0.5).float()  # Konversi nilai probabilitas menjadi label biner
        acc = (y_pred == y_train).float().mean()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

# Get final model parameters
final_params = optimizer.get_params()

# Save model parameters to file
np.save("ant_colony_model_params.npy", final_params)
