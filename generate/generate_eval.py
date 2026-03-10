from generate.mlx_generate import generate_response as generate_mlx
from generate.pytorch_generate import generate_response as generate_torch


def compare(nb_repetitions: int = 3):
    model_mlx = "mlx-community/Qwen2.5-3B-Instruct-bf16"
    model_torch = "Qwen/Qwen2.5-3B-Instruct"
    prompt = "Explique-moi la fusion nucléaire en 3 paragraphes."
    max_tokens = 5000
    torch_data = []
    mlx_data = []

    text_out = ""
    for i in range(1, nb_repetitions + 1):
        text_out += f"\n--- ITÉRATION {i} ---"
        text_out += f"\nObjectif : Générer {max_tokens * i} tokens"
        text_out += "\nPyTorch :"
        torch_data.append(generate_torch(model_torch, prompt, max_tokens * i))
        text_out += "\n" + "-"*30
        text_out += "\nMLX :"
        mlx_data.append(generate_mlx(model_mlx, prompt, max_tokens * i))
    
    print(text_out)
    text_out += "\n--- COMPARAISON FINALE ---"
    for i in range(nb_repetitions):
        text_out += f"\nItération {i+1} :"
        text_out += f"\nPyTorch - Tokens : {torch_data[i][0]}, Vitesse : {torch_data[i][1]:.2f} tokens/s"
        text_out += f"\nMLX     - Tokens : {mlx_data[i][0]}, Vitesse : {mlx_data[i][1]:.2f} tokens/s"

    print(text_out)
    text_out += "\nAnalyse :"
    for i in range(nb_repetitions):
        torch_tokens, torch_speed = torch_data[i]
        mlx_tokens, mlx_speed = mlx_data[i]
        token_diff = mlx_tokens - torch_tokens
        speed_diff = mlx_speed - torch_speed
        text_out += f"\nItération {i+1} :"
        text_out += f"\nDifférence de tokens générés (MLX - PyTorch) : {token_diff}"
        text_out += f"\nDifférence de vitesse (MLX - PyTorch) : {speed_diff:.2f} tokens/s"

    torch_mean_tokens = sum(data[0] for data in torch_data) / nb_repetitions
    torch_mean_speed = sum(data[1] for data in torch_data) / nb_repetitions
    mlx_mean_tokens = sum(data[0] for data in mlx_data) / nb_repetitions
    mlx_mean_speed = sum(data[1] for data in mlx_data) / nb_repetitions
    print(f"PyTorch - Tokens : {torch_mean_tokens:.2f}, Vitesse : {torch_mean_speed:.2f} tokens/s")
    print(f"MLX     - Tokens : {mlx_mean_tokens:.2f}, Vitesse : {mlx_mean_speed:.2f} tokens/s")

    with open("out/comparison.txt", "w") as f:
        f.write(text_out)