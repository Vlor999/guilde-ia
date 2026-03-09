from argparse import ArgumentParser
from train.mlx_training import train as train_mlx
from train.pytorch_training import train as train_pytorch
from evaluate.mlx_eval import evaluate as evaluate_mlx 
from evaluate.pytorch_eval import evaluate as evaluate_pytorch
from generate.mlx_generate import generate_response as generate_mlx
from generate.pytorch_generate import generate_response as generate_torch

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "generate", "compare", "evaluate"],
    )
    parser.add_argument("--model", type=str, default="mlx", choices=["mlx", "pytorch"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    return parser.parse_args()


def train_models_compare(
    *,
    start: int | None = None,
    end: int | None = None,
    step: int = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
):
    for epochs in range(start, end, step):
        train_mlx(epochs, batch_size, learning_rate)
        train_pytorch(epochs, batch_size, learning_rate)


def main():
    args = parse_args()
    if args.mode == "train":
        if args.model == "mlx":
            train_mlx(args.epochs, args.batch_size, args.learning_rate)
        else:
            train_pytorch(args.epochs, args.batch_size, args.learning_rate)
    elif args.mode == "compare":
        train_models_compare(start=0, end=51, step=10)
    elif args.mode == "evaluate":
        if args.model == "mlx":
            evaluate_mlx()
        else:
            evaluate_pytorch()
    elif args.mode == "generate":
        model_mlx = ""
        model_torch = ""
        prompt = "Explique-moi la fusion nucléaire en 3 paragraphes."
        max_tokens = 1000
        generate_mlx(model_mlx, prompt, max_tokens)
        generate_torch(model_torch,prompt, max_tokens)


if __name__ == "__main__":
    main()
