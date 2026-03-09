import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
from train.data_utils import load_mnist


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
    return mx.array(X), mx.array(y)

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
    for epoch in range(epochs):
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            loss, grads = loss_and_grad_fn(model, X_batch, y_batch)
            model.update(optimizer.apply_gradients(grads, model))
            mx.eval(model.parameters(), optimizer.state)

    with open(f"out/mlx/mlx_{epochs}_{batch_size}.txt", "w") as f:
        f.write(f"✅ MLX Total Time: {time.perf_counter() - start_time:.2f}s - {epochs} epochs - {batch_size} batch size - {learning_rate} learning rate")
