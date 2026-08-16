"""
Runtime strings must be plain ASCII.

Windows consoles fail on non-ASCII in two different ways, and this project has
hit both.  Characters outside cp1252 — Greek letters, ``≈``, superscript minus
— raise UnicodeEncodeError and kill the run outright; the preprocessor's
``--help`` did exactly that.  Characters inside cp1252 but outside the active
code page (often 437 or 850) survive but render as ``?`` or ``\\ufffd``, so
``0.0833 – 5 yr`` reaches the user as ``0.0833 ? 5 yr``.

ASCII avoids both. The rule enforced here:

* **Docstrings** may use whatever notation reads best — they are for people
  reading the source, and equations belong in them.
* **Plotting functions** may too — matplotlib renders Unicode properly, and
  axis labels genuinely want ``°C`` and ``W/(m²·K)``.
* **The encoding fallback itself** is exempt by name.  ``aethon.console`` holds
  the box-drawing characters it must decide *about*, and emits them only after
  probing that the stream can encode them.
* **Everything else** — anything printed, any argparse help, any exception
  message, any data string that might reach a terminal — must be ASCII.

This is a static check on source literals.  Rich draws its own borders, which
are not literals and so cannot be caught here; ``tests/test_cli_output.py``
renders the real output through a cp1252 stream to cover that half.
"""

import ast
from pathlib import Path

import pytest

_PACKAGES = ("aethon", "decay_preprocessor")

# Functions that build figure text rather than console text. Everything inside
# them is exempt, including strings assigned to a variable before being passed
# to matplotlib.
_PLOT_FUNCTION_TOKENS = ("plot", "figure", "chart", "_configure_style")

# Calls whose string arguments end up in a figure.
_PLOT_CALLS = ("title", "label", "text", "annotate", "legend", "suptitle")

# The encoding fallback in aethon/console.py has to name the characters it is
# deciding about. Exempted by name rather than by file, so any *other* string
# added to that module is still checked.
_ENCODING_FALLBACK_NAMES = ("_BOX_PROBE", "_rule_char")


def _source_files():
    root = Path(__file__).resolve().parent.parent
    for package in _PACKAGES:
        yield from sorted((root / package).rglob("*.py"))


def _strings_within(node: ast.AST) -> set:
    """Ids of every string literal anywhere inside *node*."""
    return {
        id(sub) for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    }


def _exempt_string_ids(tree: ast.AST) -> set:
    """Ids of string nodes that are docstrings or figure text."""
    exempt = set()

    for node in ast.walk(tree):
        # Docstrings
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                exempt.add(id(body[0].value))

        # Whole plotting functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lowered = node.name.lower()
            if any(token in lowered for token in _PLOT_FUNCTION_TOKENS):
                exempt |= _strings_within(node)
            if node.name in _ENCODING_FALLBACK_NAMES:
                exempt |= _strings_within(node)

        # Named constants belonging to the encoding fallback
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name in _ENCODING_FALLBACK_NAMES for name in targets):
                exempt |= _strings_within(node.value)

        # Matplotlib labelling calls, and any `label=` keyword
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if any(token in name.lower() for token in _PLOT_CALLS):
                exempt |= _strings_within(node)
            for kw in node.keywords:
                if kw.arg == "label":
                    exempt |= _strings_within(kw.value)

    return exempt


def _non_ascii(text: str) -> set:
    return {char for char in text if ord(char) > 127}


@pytest.mark.parametrize(
    "path", list(_source_files()), ids=lambda p: p.name,
)
def test_runtime_strings_are_ascii(path):
    """No runtime string may contain non-ASCII characters."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    exempt = _exempt_string_ids(tree)

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in exempt:
            continue
        bad = _non_ascii(node.value)
        if bad:
            offenders.append(
                f"  line {node.lineno}: {''.join(sorted(bad))!r} "
                f"in {node.value[:60]!r}"
            )

    assert not offenders, (
        f"{path.name} has non-ASCII runtime strings:\n"
        + "\n".join(offenders)
        + "\n\nUse ASCII where text reaches a console: 'degC' not the degree "
          "sign, 'W/(m2.K)' not superscripts, '-' not an en-dash.\n"
          "Docstrings and plotting functions are exempt."
    )
