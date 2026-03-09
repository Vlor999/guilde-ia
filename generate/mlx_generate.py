
import time
from mlx_lm import load, generate

def load_model(model_name: str):
    print(f"Chargement de MLX ({model_id})...")
    model, tokenizer = load(model_name)
    return model, tokenizer

def calcul_stats(tok, response: str, duration : float) -> None:
    tokens = tok.encode(response)
    generated_tokens = len(tokens)
    tps = generated_tokens / duration

    print("\n--- RÉSULTATS MLX ---")
    print(f"Temps total : {duration:.2f} s")
    print(f"Tokens générés : {generated_tokens}")
    print(f"Vitesse : {tps:.2f} tokens/s")
    print(response)

def generate_response(model_name: str, prompt: str, max_tokens: int) -> None:
    model, tokenizer = load_model(model_name=model_name)
    print("Génération en cours avec MLX...")
    start_time = time.perf_counter()

    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=max_tokens,
        verbose=False
    )

    end_time = time.perf_counter()
    duration = end_time - start_time
    calcul_stats(tokenizer, response, duration)
    

if __name__ == "__main__":
    model_id = "mlx-community/Phi-3-mini-4k-instruct-8bit"
    prompt = "Explique-moi la fusion nucléaire en 3 paragraphes."
    max_tokens = 1000
    generate_response(model_name=model_id, prompt=prompt, max_tokens=max_tokens)