# Contexte du projet - AutoYOLO Trainer

> **⚠️ À METTRE À JOUR À CHAQUE MODIFICATION DU PROJET.**
> Toute session (humaine ou IA) qui modifie le code, la configuration, l'architecture
> ou les dépendances doit mettre à jour ce fichier dans le même changement :
> sections concernées + une ligne dans le [Journal des modifications](#journal-des-modifications).
> Ce fichier est la source de vérité pour reprendre le travail d'une session à l'autre.

---

## Objectif

Framework d'entraînement continu de modèles **YOLO11** (Ultralytics) par
**auto-labeling + Active Learning**, compatible Windows, Python 3.12, capable de
tourner plusieurs semaines sans intervention.

Cycle : annotation manuelle initiale (Label Studio, 100-300 images) → entraînement v1 →
capture automatique des détections → tri par confiance (accepted / review / rejected) →
ré-entraînement à partir de 500 nouvelles images → promotion du modèle seulement s'il
est meilleur → versionnage des datasets et des modèles → on recommence.

Spécification d'origine : demande initiale de l'utilisateur (technologies imposées :
Ultralytics, PyTorch, OpenCV, NumPy, Pillow, Albumentations, PyYAML, Rich, Loguru, TQDM,
Hydra, Pandas, Matplotlib, Label Studio, FastAPI, Watchdog, scikit-image, ImageHash,
ONNX Runtime, TensorBoard).

## État actuel

- Toutes les fonctionnalités de la spec sont implémentées, ainsi que le dashboard FastAPI (bonus).
- **78 tests pytest passent**, `ruff check` propre.
- **Test de bout en bout réel validé** (CPU, Python 3.13, Ultralytics 8.4, torch 2.14) :
  entraînement initial → capture → auto-labeling → ré-entraînement → comparaison/non-promotion
  → rapport HTML → logs → export ONNX → stats.
- **Capture d'écran manuelle** (`scripts/capture.py`) testée en réel sous Windows : listing,
  écran entier, tous les écrans, fenêtre masquée (rendu via `PrintWindow`).
- **Non testé en réel** : appels API Label Studio (serveur jamais démarré), capture
  webcam, entraînement GPU/CUDA.
- Rien n'a encore été commité (en dehors du commit initial).

## Environnement de la machine de dev

- Windows 11, shells PowerShell + Git Bash.
- **Python 3.12 n'est pas installé** : seulement `py -3.13` et 3.11 (Store). Le code vise
  `>=3.12` ; la vérification a été faite sous 3.13 dans un venv temporaire.
- Le `.venv` du projet n'a pas encore été créé (`py -3.12 scripts\setup_env.py`).

## Architecture

```
main.py                    CLI (argparse) -> src/cli/
config/config.yaml         configuration unique (Hydra compose + validation par dataclasses)
src/
  config/                  schema.py (dataclasses), loader.py (Hydra), workspace.py (tous les chemins)
  context.py               AppContext : construit tous les composants depuis la config
  model/                   types.py (Detection, YoloBox, protocole Detector), yolo_detector.py
                           (rechargement à chaud), onnx_detector.py, registry.py (models/vN + active.json)
  datasets/                manager.py (DatasetManager), yolo_format.py, labelstudio.py (conversion pure),
                           labelstudio_client.py (REST), versioning.py (VersionRegistry vN générique),
                           augmentation.py (Albumentations + Mosaic/MixUp maison), report.py
  labeler/                 router.py (ConfidenceRouter), auto_labeler.py, watcher.py (watchdog)
  capture/                 capture_manager.py, active_learning.py, rate_limiter.py, sources.py, runner.py
  trainer/                 trainer_manager.py, backend.py (protocole TrainingBackend + Ultralytics),
                           metrics.py, report.py (HTML), scheduler.py (vérification périodique)
  dashboard/               app.py (FastAPI), stats.py (partagé avec `main.py stats`)
  cli/                     parser.py, commands.py
  utils/                   logging.py (canaux Loguru), files.py, images.py, hashing.py, html.py, lock.py
scripts/                   setup_env.py, start_labelstudio.py, import_labelstudio.py, train_initial.py,
                           capture.py (capture d'une fenêtre ou d'un écran, Windows uniquement, ctypes)
tests/                     pytest, avec FakeDetector / FakeBackend dans conftest.py (ni GPU ni réseau)
```

Dossiers de données : `datasets/{raw,current,vN,archive}`, `captures/`, `accepted/`,
`review/`, `rejected/`, `models/vN/`, `runs/`, `logs/`, `.autoyolo/` (état interne :
trainer_state.json, captures_index.csv, capture_hashes.json, training.lock, exports Label Studio).

## Décisions clés (et pourquoi)

| Décision | Raison |
|----------|--------|
| Hydra via l'API `compose`, pas `@hydra.main` | garder argparse pour la CLI et ne pas changer le répertoire de travail ; la config devient un `AppConfig` typé |
| Label Studio dans un venv séparé `.venv-labelstudio` (`requirements-labelstudio.txt`) | ses dépendances figées (Django, numpy) entrent en conflit avec Ultralytics/PyTorch |
| Images servies par le *local storage* de Label Studio (document root = racine du projet) | pas de copie ni d'upload |
| Le client LS accepte les tokens historiques (`Token`) et les PAT JWT (`Bearer` après refresh) | compatibilité avec les différentes versions de LS |
| Les snapshots `datasets/vN` utilisent des liens physiques et les écritures sont atomiques (`os.replace`) | quasi aucun espace disque ; un snapshot n'est jamais modifié par une écriture ultérieure |
| Candidat ET modèle actif évalués sur le val du nouveau snapshot | comparaison équitable alors que le dataset grossit |
| Tout candidat est enregistré dans `models/vN`, seul un meilleur devient actif | historique complet, rollback possible avec `version --promote N` |
| Tri : agrégation `min` des confiances par défaut, seuils stricts (`>`) | prudence : une seule boîte douteuse envoie l'image en review |
| Augmentation hors ligne seulement sur le train du snapshot, préfixe `aug_` | `datasets/current` reste propre ; le val n'est jamais augmenté |
| Dédoublonnage pHash 64 bits vectorisé NumPy + SSIM contre la dernière capture | rapide ; SSIM attrape les scènes statiques |
| `pending_images = len(accepted/) + imported_since_training` | un entraînement échoué ne perd pas le compteur (nouvelle tentative au prochain check) |
| Verrou fichier avec péremption de 48 h, sans test de PID | sous Windows `os.kill(pid, 0)` **tue** le processus |
| E/S d'images via `np.fromfile`/`cv2.imdecode` (`src/utils/images.py`) | `cv2.imread` échoue sur les chemins non ASCII sous Windows |
| CaptureManager écrit .txt/.json AVANT l'image | le watcher réagit à l'image et doit trouver ses fichiers associés |
| `requirements.txt` avec `--extra-index-url .../cu126` | PyTorch GPU par défaut ; `setup_env.py --cpu` pour le CPU |
| `onnxslim` ajouté aux dépendances | requis par l'export ONNX `simplify=True` d'Ultralytics |

## Conventions

- Messages CLI/logs et documentation **en français** ; identifiants, docstrings et commentaires de code en anglais.
- Typage complet, `from __future__ import annotations`, style `X | None` (ruff UP), lignes de 120 caractères max.
- Imports lourds (ultralytics, torch) faits **à la demande** dans les fonctions, pour que `stats`/`dashboard` restent rapides.
- Logs : `get_logger("training" | "capture" | "dataset")` → `logs/<canal>.log` ; ERROR+ va aussi dans `errors.log`.
- Toute nouvelle dépendance : `requirements.txt` **et** `pyproject.toml`.
- Points d'extension : protocole `Detector`, protocole `TrainingBackend`, sous-classe `FrameSource`, entrée `_FACTORIES` (augmentation).

## Commandes utiles

```powershell
py -3.12 scripts\setup_env.py [--cpu]      # crée .venv + dépendances
python scripts\start_labelstudio.py [--review]
python scripts\import_labelstudio.py [--review]
python scripts\train_initial.py            # = python main.py train
python scripts\capture.py --list           # écrans et fenêtres visibles
python scripts\capture.py --screen 1 | --window "TEXTE" [--delay 3] [-o fichier.png]
                                           # défaut : captures\manuel\AAAAMMJJ_HHMMSS.png
python main.py capture|label|retrain|stats|version|clean|run|dataset|dashboard|export
python main.py -o training.epochs=10 train # surcharge Hydra
pytest                                     # tests
ruff check .                               # lint
tensorboard --logdir runs
```

## Limites connues / pistes

- Intégration Label Studio à valider contre un vrai serveur (endpoints `/api/storages/localfiles`,
  `/api/projects/{id}/export`, variable `LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN`).
- `OnnxDetector` : nombre de détections différent d'Ultralytics à conf=0 (NMS/seuils), à vérifier sur un modèle entraîné.
- `import_labelstudio.py --review` émet des avertissements « image introuvable » pour les
  tâches déjà déplacées vers accepted/ (bruit sans conséquence).
- Les images de `review/` ne sont jamais purgées automatiquement (seul `rejected/` l'est).
- Avertissement Starlette : `httpx` déprécié pour `TestClient` (sans impact).
- `scripts/capture.py` : certaines applications (DirectX, UWP) peuvent sortir une image noire avec
  `PrintWindow` ; une fenêtre réduite n'est pas capturable. Pas de test automatisé (API Windows).

## Journal des modifications

| Date | Changement |
|------|------------|
| 2026-10-22 | Création complète du projet selon la spec (architecture, CLI, scripts, tests, README). |
| 2026-10-22 | Test de bout en bout réel avec Ultralytics validé ; ajout de la dépendance `onnxslim`. |
| 2026-10-22 | Ajout de `context.md` (ce fichier) et de `CLAUDE.md` qui l'importe automatiquement. |
| 2026-10-22 | Ajout de `scripts/capture.py` : capture d'un écran (`--screen`), de tous les écrans (`--screen 0`) ou d'une fenêtre (`--window`, rendu `PrintWindow`), listing (`--list`). |
