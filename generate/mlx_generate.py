import time
from mlx_lm import load, generate

model_id = "mlx-community/Phi-3-mini-4k-instruct-8bit"
prompt = "Explique-moi la fusion nucléaire en 3 paragraphes."
max_tokens = 1000

# 1. Chargement du modèle
print(f"Chargement de MLX ({model_id})...")
model, tokenizer = load(model_id)

# 2. Mesure de la performance
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

# 3. Calcul des stats
tokens = tokenizer.encode(response)
generated_tokens = len(tokens)
tps = generated_tokens / duration

print("\n--- RÉSULTATS MLX ---")
print(f"Temps total : {duration:.2f} s")
print(f"Tokens générés : {generated_tokens}")
print(f"Vitesse : {tps:.2f} tokens/s")
print(response)