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
- **166 tests pytest passent**, `ruff check` propre.
- **Dashboard web à menus** (`main.py dashboard`) : toutes les commandes et scripts sont lançables depuis le
  navigateur, avec sortie en direct. Testé en réel (tâche `stats` via l'API, rendu vérifié sous Chrome) ;
  les actions longues (entraînement, capture, Label Studio) n'ont pas été lancées depuis l'interface.
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
setup_env.bat              création de .venv + dépendances (double-clic ou terminal ; `--cpu` possible)
main.py                    CLI (argparse) -> src/cli/
config/config.yaml         configuration unique (Hydra compose + validation par dataclasses)
src/
  config/                  schema.py (dataclasses), loader.py (Hydra), workspace.py (tous les chemins)
  context.py               AppContext : construit tous les composants depuis la config
  model/                   catalog.py (architectures YOLO proposées), types.py (Detection, YoloBox, protocole Detector), zone.py (ZoneDetector : détection sur le carré central), yolo_detector.py
                           (rechargement à chaud), onnx_detector.py, registry.py (models/vN + active.json)
  datasets/                manager.py (DatasetManager), yolo_format.py, labelstudio.py (conversion pure),
                           labelstudio_client.py (REST), versioning.py (VersionRegistry vN générique),
                           augmentation.py (Albumentations + Mosaic/MixUp maison), report.py
  labeler/                 router.py (ConfidenceRouter), auto_labeler.py, watcher.py (watchdog)
  capture/                 capture_manager.py, active_learning.py, rate_limiter.py, sources.py (screen/window/camera/video/folder),
                           runner.py, preview.py (fenêtre de rendu des détections), windows.py (capture écran/fenêtre Win32, ctypes)
  trainer/                 trainer_manager.py, backend.py (protocole TrainingBackend + Ultralytics),
                           metrics.py, report.py (HTML), scheduler.py (vérification périodique)
  dashboard/               app.py (FastAPI, API + page), actions.py (catalogue menus/sous-menus/champs -> ligne de
                           commande), jobs.py (JobManager : un sous-processus par action, journal dans
                           .autoyolo/jobs/), static/index.html (SPA sans dépendance), stats.py (partagé avec `main.py stats`)
  cli/                     parser.py, commands.py
  utils/                   logging.py (canaux Loguru), files.py, images.py, hashing.py, html.py, lock.py
scripts/                   setup_env.py, start_labelstudio.py, import_labelstudio.py, train_initial.py,
                           capture.py (capture d'une fenêtre ou d'un écran, Windows uniquement, ctypes),
                           screenshots.py (série de captures vers datasets/raw/ pour le 1er dataset ; réutilise capture.py)
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
| Le dashboard lance les commandes en **sous-processus** (`sys.executable main.py ...`), pas en appelant les fonctions | isolation (un entraînement qui plante ne tue pas le serveur), arrêt possible, sortie capturée, comportement identique à la CLI |
| Le navigateur n'envoie que l'id d'action + valeurs ; `actions.build_command` reconstruit les arguments depuis un catalogue fixe (pas de shell, valeurs commençant par `-` refusées) | pas d'injection d'arguments |
| `POST /api/jobs` exige : `dashboard.host` en loopback, en-tête `Host` loopback, `Content-Type: application/json` | un site tiers ou un DNS rebinding ne peut pas lancer de commande ; écoute non locale = lecture seule |
| Une seule tâche par action à la fois ; les tâches en cours sont arrêtées à la fermeture du dashboard (`taskkill /T` sous Windows) | pas de processus orphelin, pas deux entraînements concurrents |
| L'aperçu (`preview`) lance le modèle sur la frame pleine résolution et ne réduit (letterbox, ratio conservé) que l'image affichée ; `run_preview` reçoit une `PreviewWindow` injectable | les boîtes restent précises, et la boucle se teste sans écran |
| Le code de capture Win32 vit dans `src/capture/windows.py` (importé par `scripts/capture.py`, `scripts/screenshots.py` et `WindowSource`) | une seule implémentation pour les scripts, la capture automatique et l'aperçu |
| Les listes déroulantes écran/fenêtre du dashboard viennent de `GET /api/screens` et `/api/windows` (champ `dynamic` du catalogue) ; mêmes garde-fous que `POST /api/jobs` (Host loopback + JSON) | les titres de fenêtres peuvent être privés |
| Zone de détection (`preview --zone`, 200 px par défaut) : `ScreenSource(zone=)` ne capture que le carré central par BitBlt (`windows.grab_region`, ~16 ms au lieu de ~62 ms : `ImageGrab` copie tout le bureau) et `ZoneDetector` garde l'échelle d'objet d'une image entière (`imgsz / plus grand côté`), l'entrée du modèle étant la zone réduite puis complétée au multiple de 32 | un crop natif agrandi montrerait des objets ~3x plus gros qu'à l'entraînement ; mesuré sur le dataset : 66 boîtes retrouvées en image entière, 58 avec la zone, 54 avec un crop natif (sur 82 au centre ; les boîtes à cheval sur le bord sont coupées) |
| La zone ne s'applique pas à `capture`/`run` | les captures sauvegardées auraient des labels incomplets (objets hors zone non détectés) |
| Sources live (écran, fenêtre, caméra) lues dans un thread (`BackgroundSource`) pour l'aperçu | la capture de l'image suivante se fait pendant l'inférence ; on traite toujours la plus récente |
| Le projet de revue Label Studio a son propre *local storage* sur `review/` (créé **sans sync**, ses tâches pré-annotées sont importées à part) | LS ne sert `/data/local-files/?d=...` que pour un dossier couvert par un stockage du projet : sans lui, images en erreur « issue loading URL from $image value » |
| Comparaison d'architectures : `train_models` / `retrain_models` entraînent chaque `--model` sur **un seul snapshot** de dataset (une seule version vN, une seule augmentation) et chaque candidat est évalué contre le modèle actif *à ce moment* | même données et même validation pour tous ; le meilleur finit actif |
| Chaque modèle enregistre `architecture`, `speed_ms`, `params_m` dans `info.json` (`benchmark` du backend ; une mesure ratée est ignorée) ; `ModelRegistry.architecture()` retrouve l'architecture des anciens modèles via `args.yaml` du run | choisir un modèle sur précision **et** vitesse ; la promotion automatique reste basée sur `evaluation.primary_metric` seule |
| Pour ajouter une action au dashboard : une entrée dans `GROUPS` (`src/dashboard/actions.py`) | le test `test_main_actions_are_accepted_by_the_real_cli_parser` vérifie qu'elle est acceptée par le parseur réel |

## Conventions

- Messages CLI/logs et documentation **en français** ; identifiants, docstrings et commentaires de code en anglais.
- Typage complet, `from __future__ import annotations`, style `X | None` (ruff UP), lignes de 120 caractères max.
- Imports lourds (ultralytics, torch) faits **à la demande** dans les fonctions, pour que `stats`/`dashboard` restent rapides.
- Logs : `get_logger("training" | "capture" | "dataset")` → `logs/<canal>.log` ; ERROR+ va aussi dans `errors.log`.
- Toute nouvelle dépendance : `requirements.txt` **et** `pyproject.toml`.
- Points d'extension : protocole `Detector`, protocole `TrainingBackend`, sous-classe `FrameSource`, entrée `_FACTORIES` (augmentation).

## Commandes utiles

```powershell
setup_env.bat [--cpu]                      # crée .venv + dépendances (lance le même travail que setup_env.py)
py -3.12 scripts\setup_env.py [--cpu]      # équivalent Python
python scripts\start_labelstudio.py [--review]
python scripts\import_labelstudio.py [--review]
python scripts\train_initial.py            # = python main.py train
python scripts\capture.py --list           # écrans et fenêtres visibles
python scripts\capture.py --screen 1 | --window "TEXTE" [--delay 3] [-o fichier.png]
                                           # défaut : captures\manuel\AAAAMMJJ_HHMMSS.png
python scripts\screenshots.py --window "TEXTE" | --screen N [--count 100] [--interval 2] [--delay 3]
                                           # série de captures -> datasets\raw\ (Ctrl+C pour arrêter)
python main.py preview [--source screen|window|camera|video|folder] [--screen N] [--window "TEXTE"] [--path P]
                      [--model poids.pt] [--conf 0.25] [--size 300] [--zone 200] [--fps N] [--duration S]
                                           # fenêtre de rendu des détections (q/Échap, + / - = seuil, z = zone/image)
python main.py train|retrain --model yolo26n.pt --model yolo11n.pt   # --model répétable = comparaison d'architectures
python main.py benchmark [--version N]     # ms/image et FPS de chaque modèle enregistré
python main.py capture|label|retrain|stats|version|clean|run|dataset|dashboard|export
python main.py -o training.epochs=10 train # surcharge Hydra
pytest                                     # tests
ruff check .                               # lint
tensorboard --logdir runs
```

## Limites connues / pistes

- Intégration Label Studio validée en réel pour : démarrage, projet principal, stockage local, import des tâches de revue,
  service des images. Reste à valider : l'annotation puis `import_labelstudio.py --review` (déplacement vers accepted/).
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
| 2026-10-23 | Ajout de `setup_env.bat` à la racine : crée `.venv` (Python 3.12 via `py`, repli sur `py -3`), installe PyTorch (cu126, ou `--cpu`) puis `requirements.txt`, et affiche la disponibilité CUDA. Non exécuté de bout en bout (installation lourde). |
| 2026-10-23 | `.venv` de dev : `torch 2.14.1+cpu` (ne chargeait pas, `WinError 126`) remplacé par `torch 2.14.1+cu126` + `torchvision 0.29.1+cu126` après `pip cache purge` (7 Go libérés). `torch.cuda.is_available()` = True. |
| 2026-10-23 | `setup_env.bat` : la vérification finale (import de torch) fait échouer le script si PyTorch ne se charge pas, au lieu d'afficher « Terminé ». |
| 2026-10-23 | Choix du modèle YOLO : catalogue `src/model/catalog.py` (YOLO26/11/v10/v8/12, n/s/m), `train`/`retrain` acceptent plusieurs `--model` (comparaison sur le même snapshot), `retrain --epochs`, nouvelle commande `benchmark`, architecture + vitesse (ms/image, FPS) dans `info.json`, `version`, rapport HTML et Modèles ▸ Historique ; dashboard : cases à cocher + action « Mesurer la vitesse ». Mesuré (même GPU, 640 px, image entière) : yolo11n ~13-14 ms/image (~70 FPS) ; yolo26n 15.0 ms contre yolo11n 13.1 ms dans un essai réel de comparaison (1 époque, espace de travail temporaire) : pas d'avantage de vitesse de YOLO26 sur ce GPU, d'où l'absence de promesse de vitesse dans le catalogue. Piège : un script de test lancé via stdin sans `if __name__ == "__main__"` bloque l'entraînement sous Windows (workers en spawn). Rapport v3 lu : non promu (mAP50-95 0.430 contre 0.460 pour v2, mais mAP50 +0.10, précision +0.19, F1 +0.11). |
| 2026-10-23 | Import de revue robuste : `import_labelstudio.py --review` plantait sur un fichier verrouillé (WinError 32) et laissait des images dans `review/` (128 déplacées sur 147, compteur 128/500). `move_with_sidecars` réessaie 5 fois sur `PermissionError`, `import_review` signale l'image en échec, continue et sort en code 1 (relancer l'import). Import relancé sur les vraies données : accepted/ 128 → 147. +2 tests. |
| 2026-10-23 | Correctif `captures_per_day` (`/api/captures-per-day`, Vue d'ensemble du dashboard) : `Series.reset_index(names=)` n'existe pas, erreur 500 dès que `captures_index.csv` existait (bug présent depuis la 1re version, jamais testé avec un vrai index). +1 test sur le format réel du CSV. |
| 2026-10-23 | Correctif projet de revue Label Studio : images non chargées (`/data/local-files/?d=review/...`) faute de stockage local sur `review/` ; `ensure_local_storage(..., sync=False)` appelé pour le projet de revue (`start_labelstudio.py --review`), nouvelle action dashboard « Envoyer les images à relire » (`--review --detach`, utilisable pendant que le serveur tourne). Vérifié sur un vrai Label Studio : stockage créé, image servie (200), 156 tâches intactes. +3 tests. |
| 2026-10-23 | Zone de détection centrale pour l'aperçu (`preview --zone`, 200 px par défaut, champ du dashboard « Zone de détection au centre ») : `ZoneDetector` (`src/model/zone.py`), `ScreenSource(zone=)` + `windows.grab_region` (BitBlt), `BackgroundSource`, `YoloDetector.predict(imgsz=)`, touche `z` (zone agrandie / image entière avec cadre), FPS lissés, limite de FPS désactivée par défaut pour les sources live. Mesuré : ~15 FPS en image entière, ~33 FPS avec la zone (écran 1680x1050, GPU). `ScreenSource` utilise BitBlt dès qu'un écran/une région est choisi. +12 tests (`tests/test_zone.py`). |
| 2026-10-23 | Choix de l'écran et de la fenêtre dans le dashboard : listes déroulantes (écrans avec résolution, fenêtres ouvertes, « Autre… » pour saisir un titre, bouton Actualiser) pour Captures d'écran, Capture automatique, Mode autonome et Tester le modèle ; nouvelles routes `/api/screens` et `/api/windows` ; option `--screen N` (`capture.source.screen`) pour `capture`/`run`/`preview` ; `ScreenSource` capture désormais tout écran (coordonnées négatives comprises) ; `show_if` accepte plusieurs valeurs. +6 tests. Testé en réel (écran 2, fenêtre de l'Explorateur). |
| 2026-10-23 | Aperçu des détections : `main.py preview` (fenêtre OpenCV 300x300, boîtes + label + confiance, `+`/`-` pour le seuil), action « Tester le modèle » du dashboard (menu Modèles), nouvelle source `window` (`WindowSource`, option `--window`, `capture.source.window`) utilisable aussi par `capture`/`run`. Code Win32 déplacé de `scripts/capture.py` vers `src/capture/windows.py` ; `capture.py --list` ne plante plus sur un titre de fenêtre non encodable. +11 tests (`tests/test_preview.py`). Testé en réel avec le modèle v2 (dossier d'images et fenêtre Chrome). |
| 2026-10-23 | Dashboard web à menus et sous-menus (Entraînement, Collecte, Label Studio, Dataset, Modèles, Maintenance, Tâches, Configuration) : `src/dashboard/{actions,jobs}.py`, `static/index.html`, nouvelles routes `/api/{catalog,jobs,models,config,captures-per-day}` ; ancienne page HTML rendue côté serveur remplacée. +40 tests (`tests/test_dashboard_ui.py`), README et `pyproject.toml` (package-data) mis à jour. |
| 2026-10-23 | Ajout de `scripts/screenshots.py` : série de captures d'une fenêtre (`--window`) ou d'un écran (`--screen`) vers `datasets/raw/` (`--count`, `--interval`, `--delay`, `-o`), pour composer le premier dataset. Réutilise les fonctions de `capture.py`. Testé sous Windows (écran entier, fenêtre introuvable, écran inexistant) ; ruff propre. |
