"""Catalog of the actions the dashboard can launch, and their safe translation to a command line.

Every action maps to an existing entry point (``main.py <command>`` or a script). The browser only
sends an action id and field values: the argument vector is rebuilt here from a whitelist, never from
free text, and no shell is involved. A value that could be read as an option (leading ``-``) is refused.
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from src.model.catalog import DEFAULT_WEIGHTS, weights_labels, weights_names

MAIN = "main.py"
OVERRIDE_PATTERN = re.compile(r"\+?[A-Za-z_][\w.]*=.*")
MAX_TEXT_LENGTH = 1024


@dataclass(frozen=True)
class Field:
    """One input of an action form.

    ``kind`` is one of text, int, float, bool, choice, multichoice, lines. A field without ``flag`` is positional.
    ``show_if`` = (other field, value or tuple of values) hides the field (and ignores it) unless that field has
    that value. ``dynamic`` ("windows" or "screens") makes the page fill a drop-down from ``/api/<dynamic>``.
    """

    name: str
    label: str
    kind: str = "text"
    flag: str | None = None
    choices: tuple[str, ...] = ()
    labels: tuple[tuple[str, str], ...] = ()  # (choice, text shown) when the value alone is not readable
    default: Any = None
    required: bool = False
    minimum: float | None = None
    placeholder: str = ""
    help: str = ""
    show_if: tuple[str, str | tuple[str, ...]] | None = None
    dynamic: str | None = None


@dataclass(frozen=True)
class Action:
    id: str
    label: str
    description: str
    script: str
    command: tuple[str, ...] = ()  # sub-command tokens (main.py only)
    fields: tuple[Field, ...] = ()
    overrides: bool = True  # accepts Hydra ``-o key=value``
    confirm: str | None = None  # asked before launch, unless the ``confirm_unless`` box is ticked
    confirm_unless: str | None = None


@dataclass(frozen=True)
class Page:
    """A menu entry rendered by the dashboard itself instead of launching a process."""

    id: str
    label: str


@dataclass(frozen=True)
class Group:
    id: str
    label: str
    items: tuple[Action | Page, ...]


@dataclass(frozen=True)
class Command:
    action: Action
    argv: list[str]  # after the Python interpreter

    @property
    def display(self) -> str:
        return subprocess.list2cmdline(self.argv)


_SOURCE = Field("source", "Source", "choice", "--source", ("screen", "window", "camera", "video", "folder"),
                default="screen", required=True)
_SCREEN = Field("screen", "Écran", "int", "--screen", minimum=0, dynamic="screens", show_if=("source", "screen"))
_WINDOW = Field("window", "Fenêtre", "text", "--window", required=True, dynamic="windows",
                show_if=("source", "window"), help="Les fenêtres réduites ne sont pas listées.")
_PATH = Field("path", "Fichier ou dossier", "text", "--path", required=True, placeholder=r"C:\videos\essai.mp4",
              show_if=("source", ("video", "folder")))
_MODELS = Field(
    "models", "Modèles à entraîner", "multichoice", "--model", weights_names(), weights_labels(),
    default=(DEFAULT_WEIGHTS,),
    help="Cochez plusieurs modèles pour les comparer : ils sont entraînés l'un après l'autre sur les mêmes images, "
         "puis mesurés (précision et vitesse). Le meilleur selon evaluation.primary_metric devient actif.",
)
_OTHER_MODEL = Field("other_model", "Autre poids (nom ou chemin .pt)", "text", "--model",
                     placeholder=r"models\v2\best.pt")
_DURATION = Field("duration", "Durée maximale (s)", "float", "--duration", minimum=1)
_DRY_RUN = Field("dry_run", "Simulation (ne supprime rien)", "bool", "--dry-run", default=True)

GROUPS: tuple[Group, ...] = (
    Group("training", "Entraînement", (
        Action("train", "Premier entraînement",
               "Entraîne un ou plusieurs modèles sur datasets/current (annotations importées de Label Studio) "
               "et les compare : précision, taille et vitesse d'inférence.",
               MAIN, ("train",), (
                   _MODELS, _OTHER_MODEL,
                   Field("epochs", "Epochs", "int", "--epochs", minimum=1, placeholder="100",
                         help="Vide = training.epochs."),
               )),
        Action("retrain", "Ré-entraînement",
               "Ré-entraîne si assez d'images ont été acceptées ; le modèle n'est promu que s'il est meilleur. "
               "Sans modèle coché, le modèle actif est affiné ; sinon chaque architecture cochée repart de ses "
               "poids pré-entraînés, pour les comparer sur les mêmes données.",
               MAIN, ("retrain",), (
                   Field("force", "Forcer (ignorer training.min_new_images)", "bool", "--force"),
                   Field("models", "Comparer ces modèles (optionnel)", "multichoice", "--model", weights_names(),
                         weights_labels(), help="Laisser vide pour continuer le modèle actif."),
                   _OTHER_MODEL,
                   Field("epochs", "Epochs", "int", "--epochs", minimum=1, placeholder="50",
                         help="Vide = training.retrain_epochs."),
               )),
    )),
    Group("collect", "Collecte", (
        Action("screenshots", "Captures d'écran",
               "Série de captures d'une fenêtre ou d'un écran, enregistrées dans datasets/raw pour le premier dataset.",
               "scripts/screenshots.py", (), (
                   Field("target", "Cible", "choice", choices=("window", "screen"), default="window", required=True),
                   Field("window", "Fenêtre", "text", "--window", required=True, dynamic="windows",
                         show_if=("target", "window"), help="Les fenêtres réduites ne sont pas listées."),
                   Field("screen", "Écran", "int", "--screen", required=True, minimum=0, dynamic="screens",
                         show_if=("target", "screen")),
                   Field("count", "Nombre de captures", "int", "--count", default=100, minimum=1),
                   Field("interval", "Pause entre captures (s)", "float", "--interval", default=2, minimum=0),
                   Field("delay", "Attente avant la 1re capture (s)", "float", "--delay", default=3, minimum=0,
                         help="Le temps de mettre la fenêtre au premier plan."),
                   Field("output", "Dossier de sortie", "text", "--output", placeholder="datasets/raw",
                         help="Vide = datasets/raw."),
               ), overrides=False),
        Action("capture", "Capture automatique",
               "Détecte avec le modèle actif et enregistre les images dans captures/ jusqu'à l'arrêt.",
               MAIN, ("capture",), (
                   _SOURCE, _SCREEN, _WINDOW, _PATH, _DURATION,
                   Field("max_frames", "Nombre maximal d'images", "int", "--max-frames", minimum=1),
               )),
        Action("label", "Auto-labeling",
               "Trie les captures en accepted / review / rejected selon la confiance du modèle.",
               MAIN, ("label",), (
                   Field("folder", "Dossier à traiter", "text", "--folder", placeholder="captures",
                         help="Vide = captures/."),
                   Field("limit", "Nombre maximal d'images", "int", "--limit", minimum=1),
                   Field("watch", "Continuer à surveiller le dossier", "bool", "--watch"),
               )),
        Action("run", "Mode autonome",
               "Capture, auto-labeling et ré-entraînement périodique, jusqu'à l'arrêt.",
               MAIN, ("run",), (_SOURCE, _SCREEN, _WINDOW, _PATH, _DURATION)),
    )),
    Group("labelstudio", "Label Studio", (
        Action("labelstudio_start", "Démarrer",
               "Installe (une fois) et démarre Label Studio, puis prépare le projet d'annotation sur datasets/raw.",
               "scripts/start_labelstudio.py", (), (
                   Field("review", "Créer aussi le projet de revue", "bool", "--review"),
                   Field("detach", "Ne pas attendre la fin du serveur", "bool", "--detach"),
               )),
        Action("labelstudio_review", "Envoyer les images à relire",
               "Envoie les images de review/ dans le projet de revue de Label Studio, pré-annotées avec les boîtes "
               "du modèle. Label Studio doit déjà tourner (sinon il est démarré en arrière-plan).",
               "scripts/start_labelstudio.py", (), (
                   Field("review", "Projet de revue", "bool", "--review", default=True),
                   Field("detach", "Ne pas attendre la fin du serveur", "bool", "--detach", default=True),
               )),
        Action("labelstudio_import", "Importer les annotations",
               "Convertit les annotations Label Studio en dataset YOLO (datasets/current).",
               "scripts/import_labelstudio.py", (), (
                   Field("review", "Projet de revue (vers accepted/)", "bool", "--review"),
                   Field("from_file", "Export JSON déjà téléchargé", "text", "--from-file",
                         placeholder=r"C:\Users\moi\Downloads\export.json",
                         help="Vide = récupération directe via l'API."),
                   Field("dedup", "Supprimer les doublons après import", "bool", "--dedup"),
               )),
    )),
    Group("dataset", "Dataset", (
        Action("dataset_report", "Rapport", "Génère le rapport HTML du dataset.", MAIN, ("dataset", "report")),
        Action("dataset_validate", "Valider", "Cherche les images ou labels invalides.", MAIN, ("dataset", "validate")),
        Action("dataset_dedup", "Doublons", "Détecte (ou supprime) les images en double.", MAIN,
               ("dataset", "dedup"), (_DRY_RUN,)),
        Action("dataset_merge", "Fusionner", "Fusionne des dossiers YOLO dans le dataset.", MAIN,
               ("dataset", "merge"), (
                   Field("sources", "Dossiers (un par ligne)", "lines", required=True,
                         placeholder=r"C:\autre_dataset"),
               )),
        Action("dataset_import", "Importer un dossier", "Importe les images d'un dossier dans le dataset.", MAIN,
               ("dataset", "import"), (
                   Field("source", "Dossier", "text", required=True, placeholder=r"C:\images"),
               )),
        Action("dataset_snapshot", "Snapshot", "Crée une version figée du dataset (datasets/vN).", MAIN,
               ("dataset", "snapshot")),
    )),
    Group("models", "Modèles", (
        Page("models", "Historique"),
        Action("preview", "Tester le modèle",
               "Ouvre une fenêtre de rendu (300x300 par défaut) qui montre les détections en direct : boîte, label "
               "et confiance. Dans la fenêtre : q ou Échap pour fermer, + / - pour régler le seuil de confiance, "
               "z pour basculer entre la zone agrandie et l'image entière.",
               MAIN, ("preview",), (
                   _SOURCE, _SCREEN, _WINDOW, _PATH,
                   Field("model", "Poids à tester (.pt)", "text", "--model", placeholder=r"models\v2\best.pt",
                         help="Vide = modèle actif."),
                   Field("conf", "Seuil de confiance", "float", "--conf", minimum=0.01, placeholder="0.25",
                         help="Vide = capture.min_confidence."),
                   Field("zone", "Zone de détection au centre (px)", "int", "--zone", default=200, minimum=0,
                         help="Seul le carré central de cette taille est capturé et analysé : bien plus de FPS. "
                              "0 = image entière. Touche z dans la fenêtre : zone agrandie / image entière."),
                   Field("size", "Taille de la fenêtre (px)", "int", "--size", default=300, minimum=100),
                   Field("fps", "Limite d'images par seconde", "float", "--fps", minimum=0.1,
                         placeholder="aucune", help="Vide = le plus vite possible (écran, fenêtre, caméra)."),
                   _DURATION,
               )),
        Action("benchmark", "Mesurer la vitesse",
               "Mesure le temps d'inférence de tous les modèles enregistrés, l'un après l'autre sur les mêmes "
               "images de validation (le tableau de l'historique affiche ms/image et FPS).",
               MAIN, ("benchmark",)),
        Action("promote", "Activer une version", "Rend le modèle vN actif (retour en arrière).", MAIN,
               ("version",), (
                   Field("version", "Numéro de version", "int", "--promote", required=True, minimum=1),
               ), confirm="Activer cette version du modèle ?"),
        Action("export", "Exporter en ONNX", "Exporte le modèle actif pour l'utiliser sans PyTorch.", MAIN,
               ("export",), (Field("dynamic", "Axes dynamiques", "bool", "--dynamic"),)),
    )),
    Group("maintenance", "Maintenance", (
        Action("clean", "Nettoyage",
               "Purge les images rejetées et les anciens runs, archive les vieux datasets.", MAIN, ("clean",), (
                   _DRY_RUN,
                   Field("dedup", "Supprimer aussi les doublons du dataset", "bool", "--dedup"),
               ), confirm="Supprimer définitivement les anciennes images rejetées et les anciens entraînements ?",
               confirm_unless="dry_run"),
        Action("stats", "Statistiques", "Affiche les statistiques du projet (sortie de main.py stats).", MAIN,
               ("stats",), (Field("fast", "Sans calcul de l'espace disque", "bool", "--fast"),)),
    )),
)


def find_action(action_id: str) -> Action:
    for group in GROUPS:
        for item in group.items:
            if isinstance(item, Action) and item.id == action_id:
                return item
    raise ValueError(f"Action inconnue: {action_id!r}")


def catalog() -> list[dict[str, Any]]:
    """JSON-ready menu: groups of actions (with their form fields) and dashboard pages."""
    return [
        {
            "id": group.id,
            "label": group.label,
            "items": [
                {"type": "action", **asdict(item)} if isinstance(item, Action) else {"type": "page", **asdict(item)}
                for item in group.items
            ],
        }
        for group in GROUPS
    ]


def _visible(field: Field, values: dict[str, Any]) -> bool:
    if field.show_if is None:
        return True
    name, expected = field.show_if
    current = values.get(name)
    return current in expected if isinstance(expected, tuple) else current == expected


def _text(field: Field, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field.label}: texte attendu")
    text = value.strip()
    if len(text) > MAX_TEXT_LENGTH or any(ch in text for ch in "\x00\r\n"):
        raise ValueError(f"{field.label}: valeur invalide")
    if text.startswith("-"):
        raise ValueError(f"{field.label}: ne peut pas commencer par « - »")
    return text


def _number(field: Field, value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{field.label}: nombre attendu")
    try:
        number = int(value) if field.kind == "int" else float(value)
    except ValueError:
        raise ValueError(f"{field.label}: nombre attendu") from None
    if not math.isfinite(number):
        raise ValueError(f"{field.label}: nombre attendu")
    if field.minimum is not None and number < field.minimum:
        raise ValueError(f"{field.label}: au moins {field.minimum:g}")
    return number


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or value == []


def _field_args(field: Field, value: Any) -> list[str]:
    """Command-line tokens of one non-empty field value."""
    if field.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{field.label}: oui/non attendu")
        return [field.flag] if value and field.flag else []
    if field.kind == "lines":
        items = value.splitlines() if isinstance(value, str) else value
        if not isinstance(items, list):
            raise ValueError(f"{field.label}: liste attendue")
        texts = [_text(field, item) for item in items if not _is_empty(item)]
        return [f"{field.flag}={text}" if field.flag else text for text in texts]
    if field.kind == "multichoice":
        if not isinstance(value, list) or any(item not in field.choices for item in value):
            raise ValueError(f"{field.label}: choix invalide")
        return [f"{field.flag}={item}" for item in dict.fromkeys(value)]
    if field.kind == "choice":
        if value not in field.choices:
            raise ValueError(f"{field.label}: choix invalide")
        if not field.flag:
            return []  # selector that only drives the form (``show_if``)
        converted: Any = value
    elif field.kind in ("int", "float"):
        converted = _number(field, value)
    else:
        converted = _text(field, value)
    return [f"{field.flag}={converted}" if field.flag else str(converted)]


def build_command(action_id: str, values: dict[str, Any], overrides: list[str]) -> Command:
    """Validate the form values and build ``[script, ...]`` (to run with the dashboard's own interpreter)."""
    action = find_action(action_id)
    known = {f.name for f in action.fields}
    if unknown := sorted(set(values) - known):
        raise ValueError(f"Champ inconnu: {', '.join(unknown)}")

    args: list[str] = []
    for field in action.fields:
        if not _visible(field, values):
            continue
        value = values.get(field.name)
        if field.kind == "bool" and value is None:
            continue
        if field.kind != "bool" and _is_empty(value):
            if field.required:
                raise ValueError(f"{field.label}: champ requis")
            continue
        args += _field_args(field, value)

    override_args: list[str] = []
    if overrides and not action.overrides:
        raise ValueError("Cette action n'accepte pas de surcharges de configuration")
    for override in overrides:
        if not isinstance(override, str) or not OVERRIDE_PATTERN.fullmatch(override.strip()):
            raise ValueError(f"Surcharge invalide (attendu CLE=VALEUR): {override!r}")
        override_args += ["-o", override.strip()]

    if action.script == MAIN:
        argv = [MAIN, *override_args, *action.command, *args]
    else:
        argv = [action.script, *args, *override_args]
    return Command(action, argv)
