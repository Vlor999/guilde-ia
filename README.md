
# guilde-ia

Bref dépôt démonstratif pour comparer des pipelines d'inférence/entraînement
avec MLX et PyTorch (exemples de génération, évaluation et entraînement).

**Présentation rapide**
- **MLX** : runtime/éco-système optimisé pour l'inférence avec modèles quantifiés
	(meilleure efficacité mémoire et accélération sur certaines plateformes).
- **mlx-lm** : client / petite librairie Python pour charger et générer
	avec les modèles MLX (interface légère pour usages rapides).
	```bash
	uv run mlx_lm.chat --model mlx-community/gpt-oss-20b-MXFP4-Q8 --max-tokens=4096
	```

**Différences importantes**
- MLX cible des modèles quantifiés et des accélérateurs spécifiques (meilleure
	efficacité mémoire).  
- PyTorch (`transformers`) utilise les poids standards (float16/bf16) et
	s'intègre au workflow PyTorch ; peut cibler `mps` sur Apple Silicon si le
	modèle tient en mémoire.

**Structure du dépôt**
- `main.py` : point d'entrée (modes `train`, `generate`, `evaluate`, `compare`).
- `generate/` : scripts de génération (`mlx_generate.py`, `pytorch_generate.py`).
- `train/` : scripts d'entraînement pour MLX et PyTorch.
- `evaluate/` : scripts d'évaluation.
- `models/` : modèles sauvegardés (mlx / torch).
- `data/` : données (ex. MNIST).

**Usage rapide**
- Générer avec MLX et PYTORCH pour comparer les résultats :

```bash
uv run python main.py --mode=generate
```
