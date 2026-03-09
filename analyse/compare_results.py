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
    coefs = {}

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

        # Calculate affine approximation (linear regression) without numpy
        if len(epochs) > 1:
            n = len(epochs)
            sum_x = sum(epochs)
            sum_y = sum(times)
            sum_xy = sum(x * y for x, y in zip(epochs, times))
            sum_x2 = sum(x**2 for x in epochs)

            denominator = n * sum_x2 - sum_x**2
            if denominator != 0:
                a = (n * sum_xy - sum_x * sum_y) / denominator
                b = (sum_y - a * sum_x) / n

                print(
                    f"[{label.upper()}] Affine function coefficients: a = {a:.4f}, b = {b:.4f}"
                )
                coefs[label.upper()] = {"a": a, "b": b}

                affine_times = [a * x + b for x in epochs]
                plt.plot(
                    epochs,
                    affine_times,
                    label=f"{label.upper()} (Fit: {a:.2f}x + {b:.2f})",
                    color=colors.get(label.lower(), None),
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                )

    # print all the rapport of coef compare to mlx
    if len(coefs) > 1:
        print("Comparaisons : ")
        for label, coef in coefs.items():
            if label != "MLX":
                print(
                    f"• [{label}] Rapport of coef compare to MLX: a = {coef['a'] / coefs['MLX']['a']:.4f}"
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
