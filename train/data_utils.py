import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist():
    """
    Loads MNIST dataset using scikit-learn, normalizes features,
    and returns it as NumPy arrays.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist["data"], mnist["target"].astype(np.int64)

    # Normalize images to [0, 1] and reshape to (N, 1, 28, 28) for PyTorch compatibility
    # MLX can handle this or reshape it if needed (N, 28, 28, 1)
    X = X.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    return X, y


def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
