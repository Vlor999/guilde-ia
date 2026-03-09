import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/Phi-3-mini-4k-instruct"
prompt = "Explique-moi la fusion nucléaire en 3 paragraphes."
max_tokens = 1000

# 1. Chargement du modèle sur le GPU (MPS)
print(f"Chargement de PyTorch ({model_id})...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype="auto", 
    device_map="mps"
)

# 2. Encodage du prompt
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

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

# 5. Calcul des stats
generated_tokens = output[0].shape[0] - inputs['input_ids'].shape[1]
tps = generated_tokens / duration

print("\n--- RÉSULTATS PYTORCH ---")
print(f"Temps total : {duration:.2f} s")
print(f"Tokens générés : {generated_tokens}")
print(f"Vitesse : {tps:.2f} tokens/s")
print(output)