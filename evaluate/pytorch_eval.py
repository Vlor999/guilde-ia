import torch
from train.pytorch_training import ConvNet
from train.data_utils import load_mnist, get_train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os


def evaluate():
    # 1. Load Data
    X, y = load_mnist()

    # Split the data into train/test using the same random state
    _, X_test, _, y_test = get_train_test_split(X, y, random_state=42)

    X_tensor = torch.from_numpy(X_test)
    y_tensor = torch.from_numpy(y_test)

    # Use a large batch size for evaluation
    test_set = TensorDataset(X_tensor, y_tensor)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # 2. Setup Device and Model
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = ConvNet().to(device)

    # 3. Load Model Weights
    weights_path = "models/torch/best_model.pth"
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights not found at {weights_path}")
        return

    print(f"Loading PyTorch model weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 4. Evaluation
    model.eval()
    correct = 0
    total = 0

    print("Evaluating model on test dataset...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Forward pass
            outputs = model(data)
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ PyTorch Test Accuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    evaluate()
