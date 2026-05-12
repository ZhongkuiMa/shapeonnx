"""Shared helpers for import architecture tests."""

__docformat__ = "restructuredtext"

import ast
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def src_root() -> Path:
    """Return the package source root.

    :return: Path to ``src/`` directory.
    """
    return Path(__file__).parent.parent.parent / "src"


def get_imports(path: Path) -> set[str]:
    """Return top-level module names imported by path.

    :param path: Path to a Python source file.
    :return: Set of top-level module names referenced in import statements.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names
