# AutoYOLO Trainer

Framework d'entraînement **continu** de modèles **YOLO11** (Ultralytics) basé sur
l'**auto-labeling** et l'**Active Learning**. Conçu pour Windows, Python 3.12, et
pour tourner plusieurs semaines sans intervention.

```
 Label Studio ──► dataset v1 ──► YOLO11n (models/v1) ──► application / capture
      ▲                                                         │
      │ revue humaine (review/)                                  ▼
      │                                    captures/ (image + .txt + .json)
      │                                                         │ AutoLabeler
      │                   ┌─────────────────────────────────────┼──────────────┐
      └──────────── review/ (85-97 %)       accepted/ (> 97 %)          rejected/ (≤ 85 %)
                                                   │ ≥ 500 nouvelles images
                                                   ▼
                          dataset vN ──► entraînement ──► comparaison ──► models/vN actif si meilleur
```

---

## Sommaire

1. [Installation](#1-installation)
2. [Création du venv](#2-création-du-venv)
3. [Installation CUDA](#3-installation-cuda)
4. [Installation de Label Studio](#4-installation-de-label-studio)
5. [Premier entraînement](#5-premier-entraînement)
6. [Ajout de nouvelles images](#6-ajout-de-nouvelles-images)
7. [Ré-entraînement](#7-ré-entraînement)
8. [Interface CLI](#8-interface-cli)
9. [Configuration](#9-configuration)
10. [Structure du projet](#10-structure-du-projet)
11. [Tests](#11-tests)
12. [FAQ](#12-faq)
13. [Bonnes pratiques](#13-bonnes-pratiques)

---

## 1. Installation

Prérequis :

| Outil | Version | Remarque |
|-------|---------|----------|
| Windows | 10 / 11 (64 bits) | Linux fonctionne aussi |
| Python | **3.12** | <https://www.python.org/downloads/> - cochez « Add python.exe to PATH » |
| Git | récent | optionnel |
| GPU NVIDIA | optionnel | fortement recommandé pour l'entraînement |

```powershell
git clone <url-du-depot> AutoYOLO
cd AutoYOLO
```

## 2. Création du venv

**Méthode automatique** (crée `.venv`, installe PyTorch CUDA puis `requirements.txt`) :

```powershell
py -3.12 scripts\setup_env.py            # GPU NVIDIA (CUDA 12.6)
py -3.12 scripts\setup_env.py --cpu      # sans GPU
.venv\Scripts\activate
```

**Méthode manuelle** :

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Si PowerShell refuse d'activer le venv :
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 3. Installation CUDA

PyTorch embarque ses propres bibliothèques CUDA/cuDNN : **il suffit d'un pilote NVIDIA
récent**, inutile d'installer le CUDA Toolkit.

1. Installez le dernier pilote : <https://www.nvidia.com/Download/index.aspx>
2. Vérifiez avec `nvidia-smi` (la version CUDA affichée doit être ≥ 12.6).
3. `requirements.txt` utilise l'index PyTorch `cu126`. Pour une autre version :
   `py -3.12 scripts\setup_env.py --cuda cu128` (ou `cu118` pour d'anciens GPU).
4. Contrôle :

   ```powershell
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

Sans GPU tout fonctionne sur CPU (entraînement beaucoup plus lent).

## 4. Installation de Label Studio

Label Studio est **installé automatiquement**, dans un venv dédié `.venv-labelstudio`
(ses dépendances épinglées - Django, numpy… - entrent souvent en conflit avec
Ultralytics/PyTorch).

1. Copiez 100 à 300 images représentatives dans `datasets/raw/`.
2. Déclarez vos classes dans `config/config.yaml` :

   ```yaml
   project:
     classes: [personne, voiture, velo]
   ```

3. Lancez :

   ```powershell
   python scripts\start_labelstudio.py
   ```

   Le script :
   - installe Label Studio au premier lancement (quelques minutes) ;
   - démarre le serveur sur <http://localhost:8080> (identifiants dans `labelstudio:` de la config) ;
   - crée le projet `AutoYOLO` avec l'interface « bounding boxes » et vos classes ;
   - branche `datasets/raw/` comme *local storage* (aucune copie, aucun upload) ;
   - affiche l'URL d'export YOLO natif.

4. Annotez dans le navigateur, puis importez :

   ```powershell
   python scripts\import_labelstudio.py
   ```

   Les annotations sont converties au format YOLO dans `datasets/current/`
   (split train/val déterministe), un rapport HTML est généré
   (`datasets/current/report.html`) et l'export brut est archivé dans `.autoyolo/labelstudio_exports/`.
   Le script est ré-exécutable : les labels corrigés sont mis à jour.

> **Token API** : par défaut le token est `labelstudio.api_token` (créé au premier
> démarrage). Changez-le via la variable d'environnement `LABEL_STUDIO_TOKEN`. Les
> *Personal Access Tokens* des versions récentes de Label Studio sont aussi acceptés.

## 5. Premier entraînement

```powershell
python scripts\train_initial.py          # ou : python main.py train
```

Paramètres par défaut (tous configurables) : **YOLO11n**, **100 epochs**, **batch auto**
(`-1`), **AMP activé**, **cache activé**.

Résultat :

```
datasets/v1/            snapshot immuable du dataset utilisé
models/v1/best.pt       meilleur modèle  (actif)
models/v1/last.pt       dernier epoch
models/v1/info.json     métriques, dataset, run
models/v1/report.html   rapport d'évaluation
runs/train_d1_.../      run Ultralytics complet (courbes, matrices de confusion)
```

Suivi en direct avec **TensorBoard** :

```powershell
tensorboard --logdir runs
```

## 6. Ajout de nouvelles images

### Capture automatique

```powershell
python main.py capture                       # capture d'écran (défaut)
python main.py capture --source camera       # webcam
python main.py capture --source video --path video.mp4
```

À chaque détection, le `CaptureManager` enregistre dans `captures/` :

```
20261022_143015_123456.png    l'image
20261022_143015_123456.txt    labels YOLO prédits
20261022_143015_123456.json   date, heure, classe, confiance, bbox, score d'active learning
```

Avant d'enregistrer, il applique :

- **limite de débit** : `capture.max_images_per_hour`, `capture.min_interval_seconds` ;
- **anti-doublons** : pHash (distance de Hamming ≤ `max_hamming_distance`) + SSIM vs dernière capture ;
- **active learning** (si `save_every_detection: false`) : score favorisant classes rares,
  petits objets, faible confiance et nouvelles positions ; un taux d'exploration garde un peu de diversité.

### Intégration dans votre propre application

```python
from src.context import AppContext

ctx = AppContext.create()
detector = ctx.detector()          # modèle actif
capture = ctx.capture_manager()

detections = detector.predict(frame)       # frame BGR (OpenCV)
capture.on_detection(frame, detections)    # sauvegarde si pertinent
```

### Tri automatique (AutoLabeler)

```powershell
python main.py label            # traite captures/
python main.py label --watch    # puis surveille le dossier (watchdog)
```

| Confiance (min des boîtes par défaut) | Destination |
|---------------------------------------|-------------|
| `> 0.97` (`confidence_accept`)        | `accepted/` |
| `0.85 < c ≤ 0.97`                     | `review/`   |
| `≤ 0.85` (`confidence_review`)        | `rejected/` |

### Revue humaine des images incertaines

```powershell
python scripts\start_labelstudio.py --review      # crée le projet "AutoYOLO - Review" pré-annoté
python scripts\import_labelstudio.py --review     # images validées -> accepted/
```

## 7. Ré-entraînement

```powershell
python main.py retrain            # si >= training.min_new_images (500) nouvelles images
python main.py retrain --force    # immédiatement
```

Cycle :

1. `accepted/` est fusionné dans `datasets/current` ;
2. snapshot `datasets/vN` (liens physiques NTFS : quasi aucun espace disque) ;
3. augmentation hors ligne (Albumentations + Mosaic + MixUp) sur le **train** du snapshot ;
4. entraînement (fine-tuning du modèle actif par défaut) ;
5. **le candidat et le modèle actif sont évalués sur le même jeu de validation** ;
6. comparaison (mAP50, mAP50-95, Precision, Recall, F1 ; métrique principale configurable) ;
7. le candidat est enregistré dans `models/vM` ; il devient **actif seulement s'il est meilleur** ;
8. rapport HTML `models/vM/report.html`.

### Mode autonome (plusieurs semaines sans intervention)

```powershell
python main.py run
```

Capture + auto-labeling en continu + vérification périodique
(`training.check_interval_minutes`) du seuil de ré-entraînement. Quand un nouveau
modèle est promu, il est **rechargé à chaud** dans la boucle de capture. Une erreur
sur une image n'arrête jamais la boucle ; un verrou empêche deux entraînements simultanés.

Pour un lancement au démarrage de Windows, créez une tâche planifiée :

```powershell
schtasks /Create /SC ONLOGON /TN AutoYOLO /TR "C:\chemin\AutoYOLO\.venv\Scripts\python.exe C:\chemin\AutoYOLO\main.py run"
```

### Rollback

```powershell
python main.py version              # historique des modèles et datasets
python main.py version --promote 3  # réactiver models/v3
```

## 8. Interface CLI

| Commande | Rôle |
|----------|------|
| `python main.py train` | premier entraînement |
| `python main.py capture` | capture automatique |
| `python main.py label` | auto-labeling / tri |
| `python main.py retrain` | ré-entraînement conditionnel |
| `python main.py stats` | statistiques du projet |
| `python main.py version` | versions modèles/datasets, `--promote N` |
| `python main.py clean` | purge `rejected/` ancien, vieux runs, archive les vieux datasets (`--dry-run`, `--dedup`) |
| `python main.py run` | mode autonome complet |
| `python main.py dataset report\|validate\|dedup\|merge\|import\|snapshot` | outils dataset |
| `python main.py dashboard` | tableau de bord web <http://127.0.0.1:8000> |
| `python main.py export` | export ONNX du modèle actif |

Toute valeur de configuration peut être surchargée : `python main.py -o training.epochs=20 -o training.device=0 train`.

### Tableau de bord

`python main.py dashboard` affiche (rafraîchi toutes les 30 s) : nombre d'images et
d'annotations, progression du dataset, derniers entraînements, meilleure mAP, modèle
actif, espace disque, captures, images en attente de validation. JSON brut : `/api/stats`.

## 9. Configuration

Tout est dans [`config/config.yaml`](config/config.yaml) (chargé par Hydra, validé par
les dataclasses de [`src/config/schema.py`](src/config/schema.py)). Sections :

| Section | Contenu |
|---------|---------|
| `project` | nom, liste des classes |
| `paths` | emplacement de chaque dossier |
| `capture` | source (screen/camera/video/folder), débit, format |
| `active_learning` | poids des signaux, seuils, dédoublonnage pHash/SSIM |
| `labeling` | seuils accept/review, agrégation `min`/`mean`, politique image vide |
| `training` | modèle de base, epochs, batch, AMP, cache, seuil de ré-entraînement |
| `evaluation` | métrique principale, gain minimal pour promouvoir |
| `augmentation` | chaque transformation (enabled, p, params), Mosaic, MixUp, augmentations Ultralytics |
| `labelstudio` | URL, identifiants, token, titres de projets |
| `dashboard`, `logging`, `clean` | serveur web, rotation des logs, rétention |

## 10. Structure du projet

```
AutoYOLO/
├── main.py                  point d'entrée CLI
├── config/config.yaml       configuration unique
├── datasets/
│   ├── raw/                 images à annoter (Label Studio)
│   ├── current/             dataset de travail (images/, labels/, data.yaml)
│   ├── v1/ v2/ ...          snapshots immuables utilisés pour chaque entraînement
│   └── archive/             anciens snapshots zippés (main.py clean)
├── captures/                captures brutes en attente de tri
├── accepted/ review/ rejected/   résultat du tri par confiance
├── models/v1/ v2/ ...       best.pt, last.pt, info.json, report.html (+ active.json)
├── runs/                    runs Ultralytics + TensorBoard
├── logs/                    training.log, capture.log, dataset.log, errors.log
├── .autoyolo/               état interne (compteurs, index CSV des captures, hash)
├── scripts/
│   ├── setup_env.py         création du venv
│   ├── start_labelstudio.py installation/démarrage Label Studio + projet
│   ├── import_labelstudio.py export Label Studio -> YOLO
│   └── train_initial.py     premier entraînement
├── src/
│   ├── config/              schéma typé, chargeur Hydra, Workspace (chemins)
│   ├── capture/             CaptureManager, active learning, sources, boucle, rate limiter
│   ├── labeler/             AutoLabeler, ConfidenceRouter, watcher (watchdog)
│   ├── datasets/            DatasetManager, format YOLO, Label Studio, versioning, augmentation
│   ├── model/               types (Detection, YoloBox), détecteurs YOLO/ONNX, ModelRegistry
│   ├── trainer/             TrainerManager, backend Ultralytics, métriques, rapport, scheduler
│   ├── dashboard/           FastAPI + statistiques
│   ├── cli/                 parseur et commandes
│   ├── utils/               logging, fichiers, images, hash, HTML, verrou
│   └── context.py           assemblage des composants depuis la config
└── tests/                   pytest
```

Points d'extension :

- **Autre détecteur** : implémentez le protocole `Detector` (`predict(image) -> list[Detection]`), cf. `OnnxDetector`.
- **Autre framework d'entraînement** : implémentez `TrainingBackend` (`train`, `evaluate`).
- **Autre source d'images** : sous-classez `FrameSource`.
- **Nouvelle augmentation** : ajoutez une entrée dans `_FACTORIES` (`src/datasets/augmentation.py`).

## 11. Tests

```powershell
pytest
```

Les tests couvrent DatasetManager, AutoLabeler, CaptureManager, TrainerManager, la
conversion Label Studio ↔ YOLO, le format YOLO, la suppression des doublons,
l'augmentation, la configuration et le dashboard. Ils n'ont besoin ni de GPU, ni de
poids YOLO, ni de Label Studio (détecteur et backend d'entraînement factices).

## 12. FAQ

**Les accents s'affichent mal dans la console.**
Utilisez Windows Terminal, ou exécutez `chcp 65001` et `set PYTHONUTF8=1`.

**`RuntimeError: ... freeze_support` / workers qui plantent sous Windows.**
Les scripts ont déjà la garde `if __name__ == "__main__"`. Si le problème persiste,
réduisez `training.workers` (ex. `-o training.workers=0`).

**CUDA out of memory.**
Le batch auto (`-1`) vise ~60 % de la VRAM. Fixez un batch (`-o training.batch=8`),
réduisez `training.imgsz` ou passez `training.cache=disk` / `false`.

**Le cache RAM consomme trop de mémoire.**
`training.cache: disk` (cache sur disque) ou `false`.

**Label Studio n'affiche pas les images.**
Les images doivent être sous le dossier du projet (document root) ; relancez
`scripts/start_labelstudio.py` qui active `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED`.

**Authentification Label Studio refusée.**
Le token est créé au *premier* démarrage. Si vous l'avez changé ensuite, récupérez
le token dans *Account & Settings* de Label Studio et exportez `LABEL_STUDIO_TOKEN`.

**Le nouveau modèle n'est jamais promu.**
Regardez `models/vN/report.html`. Vérifiez la qualité de `accepted/` (seuil trop bas ?),
ou baissez `evaluation.min_improvement`. Le modèle reste dans l'historique.

**Trop d'images quasi identiques.**
Diminuez `capture.max_images_per_hour`, augmentez `active_learning.dedup.max_hamming_distance`,
baissez `ssim_threshold`, ou passez `capture.save_every_detection: false`.

**Le disque se remplit.**
`python main.py clean` (programmable chaque nuit) ; les snapshots utilisent des liens physiques.

**Déployer sans PyTorch ?**
`python main.py export` produit `models/vN/best.onnx`, utilisable avec `src.model.onnx_detector.OnnxDetector`.

## 13. Bonnes pratiques

- **Dataset initial varié** (éclairages, angles, distances, arrière-plans) plutôt que volumineux.
- Incluez **5-10 % d'images sans objet** (fond) pour limiter les faux positifs.
- Gardez des **seuils prudents** au début (`confidence_accept` ≥ 0.97) : l'auto-labeling
  amplifie les erreurs du modèle ; revoyez régulièrement `review/`.
- Contrôlez de temps en temps un échantillon de `accepted/` : le jeu de validation
  grossit aussi avec ces images.
- Utilisez la **métrique `map50_95`** pour la promotion (plus exigeante sur la précision des boîtes).
- Ne modifiez jamais `datasets/vN` ni `models/vN` à la main : ils constituent l'historique.
- Programmez `python main.py clean` et surveillez `logs/errors.log`.
- Versionnez `config/config.yaml` avec Git pour tracer vos réglages.
