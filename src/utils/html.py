"""Tiny HTML helpers shared by the reports and the dashboard (no template engine)."""

from __future__ import annotations

import base64
import io
from collections.abc import Iterable, Sequence
from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_CSS = """
:root { --bg:#f7f7f9; --fg:#1d1d24; --card:#fff; --muted:#666; --accent:#2b6cb0; --ok:#2f855a; --bad:#c53030; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16161c; --fg:#e8e8ee; --card:#20202a; --muted:#9a9aa8; --accent:#63b3ed; --ok:#68d391; --bad:#fc8181; }
}
body { font-family: system-ui, Segoe UI, sans-serif; background:var(--bg); color:var(--fg); margin:0; padding:24px; }
h1 { margin-top:0 } .muted { color:var(--muted) }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px; margin:16px 0; }
.card { background:var(--card); border-radius:10px; padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,.08); }
.card .value { font-size:1.6em; font-weight:600; color:var(--accent); }
table { border-collapse:collapse; width:100%; background:var(--card); border-radius:10px; overflow:hidden; }
th, td { padding:8px 12px; text-align:left; border-bottom:1px solid rgba(127,127,127,.2); }
th { font-weight:600 } .ok { color:var(--ok) } .bad { color:var(--bad) }
img.chart { max-width:100%; background:#fff; border-radius:10px; }
"""


def page(title: str, body: str, refresh_seconds: int | None = None) -> str:
    refresh = f'<meta http-equiv="refresh" content="{refresh_seconds}">' if refresh_seconds else ""
    return (
        f'<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">{refresh}'
        f"<title>{escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><h1>{escape(title)}</h1>{body}</body></html>"
    )


def cards(items: Iterable[tuple[str, Any]]) -> str:
    inner = "".join(
        f'<div class="card"><div class="muted">{escape(label)}</div>'
        f'<div class="value">{escape(str(value))}</div></div>'
        for label, value in items
    )
    return f'<div class="grid">{inner}</div>'


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], raw_cells: bool = False) -> str:
    """Render a table; ``raw_cells=True`` allows pre-formatted HTML in cells."""
    fmt = (lambda v: str(v)) if raw_cells else (lambda v: escape(str(v)))
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def figure_to_img(fig: Figure) -> str:
    """Embed a Matplotlib figure as a base64 PNG ``<img>`` tag."""
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img class="chart" src="data:image/png;base64,{data}">'
