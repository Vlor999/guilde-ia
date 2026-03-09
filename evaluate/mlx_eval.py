import mlx.core as mx
import os
from train.mlx_training import ConvNet
from train.data_utils import load_mnist, get_train_test_split


def evaluate():
    # 1. Load Data
    X, y = load_mnist()

    # Adjust axes for MLX (N, H, W, C)
    X = X.transpose(0, 2, 3, 1)

    # Split the data into train/test using the same random state
    _, X_test, _, y_test = get_train_test_split(X, y, random_state=42)

    X_test_mx = mx.array(X_test)
    y_test_mx = mx.array(y_test)

    # 2. Setup Model
    model = ConvNet()

    # 3. Load Model Weights
    weights_path = "models/mlx/best_model.safetensors"
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights not found at {weights_path}")
        return

    print(f"Loading MLX model weights from {weights_path}...")
    model.load_weights(weights_path)
    mx.eval(model.parameters())

    # 4. Evaluation
    print("Evaluating model on test dataset...")

    # Evaluate in batches to avoid OOM issues on large test sets
    batch_size = 256
    correct = 0
    total = 0

    for i in range(0, X_test_mx.shape[0], batch_size):
        X_batch = X_test_mx[i : i + batch_size]
        y_batch = y_test_mx[i : i + batch_size]

        # Forward pass
        logits = model(X_batch)

        # Get predictions
        predictions = mx.argmax(logits, axis=1)

        # Count correct predictions
        correct += mx.sum(predictions == y_batch).item()
        total += y_batch.shape[0]

    accuracy = 100 * correct / total
    print(f"✅ MLX Test Accuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    evaluate()
