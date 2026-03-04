"""
Configuration loader for solver_config.yaml.

Parses the user-facing YAML file and returns a structured configuration
object with callable material property functions reconstructed from
expression strings and decay coefficients.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict


def _make_expression(expr_str: str):
    """
    Build a vectorised callable from a Python expression string.

    The expression is evaluated with ``T`` as temperature in Kelvin
    and ``np`` (numpy) available.

    Examples::

        _make_expression("500.0")                       # constant
        _make_expression("500.0 + 0.5*T")               # linear
        _make_expression("200.0 * T**0.35")             # power law
        _make_expression("np.interp(T, [...], [...])")  # piecewise
    """
    _ns = {"np": np}

    def expr_func(T):
        local = {**_ns, "T": np.asarray(T, dtype=float)}
        return eval(expr_str, {"__builtins__": {}}, local)  # noqa: S307

    return expr_func


def _make_decay(terms: list):
    """
    Build ``Q(t) = Σ Aᵢ · exp(−λᵢ · t)`` from ``[[A1, λ1], [A2, λ2], ...]``.

    Returns a callable: ``Q(t_years) → float`` giving specific decay power [W/kg].
    """
    arr = np.array(terms, dtype=float)

    def decay(t):
        return np.sum(arr[:, 0] * np.exp(-arr[:, 1] * t))

    return decay


def load_config(yaml_path: str = None) -> Dict[str, Any]:
    """
    Load and parse ``solver_config.yaml``.

    Parameters
    ----------
    yaml_path : str or Path, optional
        Path to config file. Defaults to ``solver_config.yaml`` in the current
        working directory.

    Returns
    -------
    dict
        Structured configuration with keys:

        - ``waste_form_name`` (str) — label used in output filenames
        - ``waste_form`` (dict) — ``rho_base`` (float), ``decay``/``cp``/``k`` (callables)
        - ``centerline_limit_C`` (float)
        - ``safety_factor`` (float)
        - ``surface_limits_C`` (dict)
        - ``ambient_temp_C`` (float)
        - ``h_passive`` (float)
        - ``radii_min``, ``radii_max``, ``radii_steps`` (float / int)
        - ``loadings_pct`` (list of float)
        - ``max_h_active``, ``max_cooling_years`` (float)
        - ``nodes``, ``max_years``, ``cooling_months`` (int / float)
    """
    if yaml_path is None:
        yaml_path = Path.cwd() / "solver_config.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    wf_raw = raw["waste_form"]

    waste_form = {
        "rho_base": float(wf_raw["rho_base"]),
        "decay":    _make_decay(wf_raw["decay_terms"]),
        "cp":       _make_expression(str(wf_raw["cp"])),
        "k":        _make_expression(str(wf_raw["k"])),
    }

    return {
        "waste_form_name": str(raw.get("waste_form_name", "WasteForm")),
        "waste_form": waste_form,
        "centerline_limit_C": float(raw["centerline_limit_C"]),
        "safety_factor": float(raw["safety_factor"]),
        "surface_limits_C": {k: float(v) for k, v in raw["surface_limits_C"].items()},
        "ambient_temp_C": float(raw["ambient_temp_C"]),
        "h_passive": float(raw["h_passive"]),
        "radii_min": float(raw["radii_min"]),
        "radii_max": float(raw["radii_max"]),
        "radii_steps": int(raw["radii_steps"]),
        "loadings_pct": [float(x) for x in raw["loadings_pct"]],
        "max_h_active": float(raw.get("max_h_active", np.inf)),
        "max_cooling_years": float(raw.get("max_cooling_years", np.inf)),
        "nodes": int(raw["nodes"]),
        "max_years": float(raw["max_years"]),
        "cooling_months": float(raw["cooling_months"]),
    }
