import re
import os
import matplotlib.pyplot as plt
from pathlib import Path


def found_all_files(directory: str | Path) -> list[str]:
    abs_path_files = []
    if not os.path.exists(directory):
        return []
    files_or_dirs = os.listdir(directory)
    for obj in files_or_dirs:
        abs_path = os.path.abspath(Path(directory) / obj)
        if os.path.isfile(abs_path):
            abs_path_files.append(abs_path)
    return abs_path_files


def found_directories(directory: str | Path) -> dict[str, str]:
    abs_path_dir = {}
    if not os.path.exists(directory):
        return {}
    files_or_dirs = os.listdir(directory)
    for obj in files_or_dirs:
        abs_path = os.path.abspath(Path(directory) / obj)
        if os.path.isdir(abs_path):
            abs_path_dir[obj] = abs_path
    return abs_path_dir


def get_data(filename: str | Path) -> dict[str, float]:
    data = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "Total Time" not in line:
                continue

            # Extract values based on keywords
            time_match = re.search(r"Total Time:\s*([\d\.]+)s", line)
            epochs_match = re.search(r"(\d+)\s*epochs", line)
            batch_match = re.search(r"(\d+)\s*batch size", line)
            lr_match = re.search(r"([\d\.]+)\s*learning rate", line)

            if time_match:
                data["time"] = float(time_match.group(1))
            if epochs_match:
                data["epochs"] = int(epochs_match.group(1))
            if batch_match:
                data["batch_size"] = int(batch_match.group(1))
            if lr_match:
                data["lr"] = float(lr_match.group(1))

    return data


def compare_data(all_data):
    plt.style.use("bmh")
    plt.figure(figsize=(10, 6))

    colors = {"mlx": "#007AFF", "torch": "#FF3B30"}
    markers = {"mlx": "o", "torch": "s"}

    for label, data_list in all_data.items():
        valid_data = [d for d in data_list if d]
        sorted_data = sorted(valid_data, key=lambda x: x["epochs"])

        epochs = [d["epochs"] for d in sorted_data]
        times = [d["time"] for d in sorted_data]

        plt.plot(
            epochs,
            times,
            label=f"{label.upper()}",
            color=colors.get(label.lower(), None),
            marker=markers.get(label.lower(), "x"),
            linewidth=2,
            markersize=8,
        )

    plt.title(
        "Training Performance Comparison: MLX vs PyTorch",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Number of Epochs", fontsize=12)
    plt.ylabel("Total Time (seconds)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)

    # Add a watermark or signature style
    plt.tight_layout()

    output_path = "analyse/comparison_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Graph saved to {output_path}")


def main():
    BASE_DIRECTORY = "out"
    dirs = found_directories(BASE_DIRECTORY)

    all_data = {}
    for dir_name, abs_path_dir in dirs.items():
        files = found_all_files(abs_path_dir)
        data_dir = []
        for file in files:
            parsed = get_data(file)
            if parsed:
                data_dir.append(parsed)
        all_data[dir_name] = data_dir

    if all_data:
        compare_data(all_data)
    else:
        print("No data found to compare.")


if __name__ == "__main__":
    main()
