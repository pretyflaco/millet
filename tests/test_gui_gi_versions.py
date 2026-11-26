"""Regression test for #26: GTK3/GTK4-split namespaces must be version-pinned.

Parses ``millet/gui.py`` instead of importing it, so it runs without
PyGObject or a display.
"""

from __future__ import annotations

import ast
from pathlib import Path

GUI_SRC = Path(__file__).resolve().parent.parent / "millet" / "gui.py"

# Namespaces that exist in both GTK3 and GTK4 flavours.
SPLIT_NAMESPACES = {"Gtk", "Gdk"}


def test_gtk_namespaces_pinned_to_3_before_import():
    tree = ast.parse(GUI_SRC.read_text(encoding="utf-8"))

    pinned: dict[str, tuple[int, str]] = {}  # namespace -> (line, version)
    imported: dict[str, int] = {}  # namespace -> first import line
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "require_version"
            and len(node.args) >= 2
            and all(isinstance(a, ast.Constant) for a in node.args[:2])
        ):
            pinned[node.args[0].value] = (node.lineno, node.args[1].value)
        elif isinstance(node, ast.ImportFrom) and node.module == "gi.repository":
            for alias in node.names:
                imported.setdefault(alias.name, node.lineno)

    for ns in SPLIT_NAMESPACES & imported.keys():
        assert ns in pinned, f"{ns} imported without gi.require_version"
        line, version = pinned[ns]
        assert version == "3.0", f"{ns} pinned to {version}, expected 3.0"
        assert line < imported[ns], f"{ns} pinned after it is imported"
