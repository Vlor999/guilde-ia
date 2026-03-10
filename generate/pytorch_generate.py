import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen2.5-3B-Instruct : ~6 Go en float16, compatible MPS (Apple Silicon GPU).
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(model_name: str):
    # 1. Chargement du modèle
    print(f"Chargement de PyTorch ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    return tokenizer, model

def calcul_stats(tokenizer, inputs, response, duration):

    # Décoder uniquement les tokens générés (hors prompt)
    input_len = inputs['input_ids'].shape[1]
    generated_ids = response[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = len(generated_ids)
    tps = generated_tokens / duration

    print("\n--- RÉSULTATS PYTORCH ---")
    print(f"Temps total : {duration:.2f} s")
    print(f"Tokens générés : {generated_tokens}")
    print(f"Vitesse : {tps:.2f} tokens/s")
    return generated_tokens, tps

def generate_response(model_name: str, prompt: str, max_tokens: int) -> tuple[int, float]:
    # 2. Encodage du prompt
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # 3. Warm-up (pour éviter les délais d'initialisation dans la mesure)
    _ = model.generate(**inputs, max_new_tokens=10)

    # 4. Mesure de la performance
    print("Génération en cours avec PyTorch...")
    start_time = time.perf_counter()

    output = model.generate(
        **inputs, 
        max_new_tokens=max_tokens,
        do_sample=True,
    )

    end_time = time.perf_counter()
    duration = end_time - start_time
    generated_tokens, tps = calcul_stats(tokenizer, inputs, output, duration)
    return generated_tokens, tps


if __name__ == "__main__":
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    prompt = "Explique-moi la fusion nucléaire en 3 paragraphes."
    max_tokens = 1000
    generate_response(model_name, prompt, max_tokens)