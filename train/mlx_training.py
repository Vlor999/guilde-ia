import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import os
from train.data_utils import load_mnist, get_train_test_split


# 1. CNN Architecture
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


def get_mnist_mlx():
    X, y = load_mnist()
    # MLX Conv2d usually expects (N, H, W, C) or (N, C, H, W) depending on config
    # The existing model uses (N, 28, 28, 1)
    X = X.transpose(0, 2, 3, 1)  # (N, 1, 28, 28) -> (N, 28, 28, 1)

    X_train, X_test, y_train, y_test = get_train_test_split(X, y, random_state=42)
    return mx.array(X_train), mx.array(y_train)


def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")


def train(epochs: int, batch_size: int, learning_rate: float):
    model = ConvNet()
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=learning_rate)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    X, y = get_mnist_mlx()

    print(f"Start MLX - {epochs} - {batch_size} - {learning_rate}")
    start_time = time.perf_counter()
    best_loss = float("inf")

    os.makedirs("out/mlx", exist_ok=True)
    os.makedirs("models/mlx", exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            loss, grads = loss_and_grad_fn(model, X_batch, y_batch)
            model.update(optimizer.apply_gradients(grads, model))
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_weights("models/mlx/best_model.safetensors")

    with open(f"out/mlx/mlx_{epochs}_{batch_size}.txt", "w") as f:
        f.write(
            f"✅ MLX Total Time: {time.perf_counter() - start_time:.2f}s - {epochs} epochs - {batch_size} batch size - {learning_rate} learning rate - best train loss: {best_loss:.4f}"
        )
