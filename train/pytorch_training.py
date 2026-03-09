import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from train.data_utils import load_mnist, get_train_test_split
from torch.utils.data import DataLoader, TensorDataset


# 1. CNN Architecture
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.features(x)

def train(epochs: int, batch_size: int, learning_rate: float):
    X, y = load_mnist()

    X_train, X_test, y_train, y_test = get_train_test_split(X, y, random_state=42)

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    train_set = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 3. Training Setup
    device = torch.device("mps")
    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    print(f"Start pytorch - {epochs} - {batch_size} - {learning_rate}")
    start_time = time.perf_counter()
    best_loss = float("inf")

    os.makedirs("out/torch", exist_ok=True)
    os.makedirs("models/torch", exist_ok=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save the model state dictionary
            torch.save(model.state_dict(), "models/torch/best_model.pth")

    torch.mps.synchronize()
    with open(f"out/torch/torch_{epochs}_{batch_size}.txt", "w") as f:
        f.write(
            f"✅ PyTorch Total Time: {time.perf_counter() - start_time:.2f}s - {epochs} epochs - {batch_size} batch size - {learning_rate} learning rate - best train loss: {best_loss:.4f}"
        )
