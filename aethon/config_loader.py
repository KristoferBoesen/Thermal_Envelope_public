"""
Configuration loader for solver_config.yaml.

Parses the user-facing YAML file and returns a structured configuration
object with callable material property functions reconstructed from
expression strings and decay coefficients.

Two concerns are kept separate in the YAML and merged here:

**Material** — the immobilisation matrix (``rho_base``, ``k(T)``, ``cp(T)``,
and its devitrification limit).  Chosen once per run from a named library.

**Waste stream** — the decay curve and campaign mass, which come from the
decay preprocessor and are independent of what matrix the waste is put into.

Note there is no global ambient temperature.  The active facility's ambient is
carried by the chosen cooling archetype, because an HTC and the temperature it
works against only mean something together.  Only the passive phases — the
interim store and the repository — have a site-wide ambient, and that is a
worst-case design basis rather than an expected value.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional


def _make_expression(expr_str: str):
    """
    Build a vectorised callable from a Python expression string.

    The expression is evaluated with ``T`` as temperature in Kelvin
    and ``np`` (numpy) available.

    The result is always broadcast to the shape of ``T``.  Without this a
    constant expression such as ``"1.2"`` would return a bare scalar, and
    callers that index per-node (``k[-1]`` in the surface boundary condition)
    would fail.

    Examples::

        _make_expression("500.0")                       # constant
        _make_expression("500.0 + 0.5*T")               # linear
        _make_expression("200.0 * T**0.35")             # power law
        _make_expression("np.interp(T, [...], [...])")  # piecewise
        _make_expression("935.56 + 0.38953*T - 24617000/T**2")  # Shomate-style
    """
    _ns = {"np": np}

    def expr_func(T):
        T_arr = np.asarray(T, dtype=float)
        local = {**_ns, "T": T_arr}
        result = eval(expr_str, {"__builtins__": {}}, local)  # noqa: S307
        return np.broadcast_to(np.asarray(result, dtype=float), T_arr.shape)

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


def _resolve_waste_source(raw: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    """
    Load the preprocessor's ``waste_source.yaml`` if the config references one.

    Relative paths resolve against the directory holding the config file, not
    the process working directory, so a config can be run from anywhere.

    Returns an empty dict when no ``waste_source`` key is set.  Callers should
    treat that as "using inline ``decay_terms``" and say so to the user — a
    silently substituted decay curve is the most damaging mistake available
    here, since it produces plausible results for waste you do not have.
    """
    src = raw.get("waste_source")
    if not src:
        return {}

    src_path = Path(src)
    if not src_path.is_absolute():
        src_path = config_dir / src_path

    if not src_path.exists():
        raise FileNotFoundError(
            f"waste_source file not found: {src_path}\n"
            "Run the decay preprocessor to generate it, or remove the "
            "'waste_source' key to use inline decay_terms."
        )

    with open(src_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    payload["_path"] = src_path
    return payload


def _require_range(
    raw: Dict[str, Any], key: str, steps: bool = False,
) -> Dict[str, Any]:
    """
    Read a ``{min, max, steps}`` block, failing with the shape it expected.

    Ranges are the one place in the file where a user is likely to guess a
    layout, so a wrong guess should say what the right one looks like rather
    than raising a bare ``KeyError`` naming half a key.
    """
    wanted = ("min", "max", "steps") if steps else ("min", "max")
    block = raw.get(key)

    if not isinstance(block, dict) or any(k not in block for k in wanted):
        example = "\n".join(f"      {k}: ..." for k in wanted)
        raise KeyError(
            f"'{key}' must be a block with "
            f"{', '.join(wanted)}:\n\n    {key}:\n{example}\n"
        )

    if float(block["min"]) > float(block["max"]):
        raise ValueError(
            f"'{key}' has min ({block['min']:g}) greater than max "
            f"({block['max']:g})."
        )
    if steps and int(block["steps"]) < 1:
        raise ValueError(
            f"'{key}.steps' must be at least 1, got {block['steps']}."
        )

    return block


def _resolve_material(
    raw: Dict[str, Any], decay_terms: list,
) -> tuple:
    """
    Build the ``waste_form`` property dict and resolve the centreline limit.

    Prefers the ``materials`` library selected by the ``material`` key.  Falls
    back to the legacy singular ``waste_form`` block so pre-existing configs
    keep working.

    Returns
    -------
    tuple of (waste_form_dict, material_name, centerline_limit_C, spec, terms)
        ``spec`` is the raw material entry and ``terms`` the raw decay pairs,
        both carried through so a run can be recorded exactly as it was
        configured — the callables built here cannot be serialised back.
    """
    materials = raw.get("materials")

    if materials:
        name = raw.get("material")
        if name is None:
            if len(materials) != 1:
                available = ", ".join(sorted(materials))
                raise ValueError(
                    f"Multiple materials defined ({available}). "
                    "Set 'material:' in the config or pass --material."
                )
            name = next(iter(materials))
        if name not in materials:
            available = ", ".join(sorted(materials))
            raise ValueError(
                f"Material '{name}' not found in config. Available: {available}"
            )
        mat = materials[name]
        limit = mat.get("centerline_limit_C")
    else:
        # Legacy layout: a single inline 'waste_form' block
        mat = raw["waste_form"]
        name = str(raw.get("waste_form_name", "WasteForm"))
        limit = mat.get("centerline_limit_C")
        if decay_terms is None:
            decay_terms = mat["decay_terms"]

    if decay_terms is None:
        raise ValueError(
            "No decay_terms found. Provide them inline, inside the selected "
            "material, or via a 'waste_source' file from the preprocessor."
        )

    waste_form = {
        "rho_base": float(mat["rho_base"]),
        "decay": _make_decay(decay_terms),
        "cp": _make_expression(str(mat["cp"])),
        "k": _make_expression(str(mat["k"])),
    }
    spec = {
        "rho_base": float(mat["rho_base"]),
        "k": str(mat["k"]),
        "cp": str(mat["cp"]),
    }
    if limit is not None:
        spec["centerline_limit_C"] = float(limit)

    terms = [[float(A), float(lam)] for A, lam in decay_terms]
    return waste_form, name, limit, spec, terms


def _resolve_surface_limits(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Repository geology → surface temperature limit [°C].

    Accepts the structured ``repositories`` block or the legacy flat
    ``surface_limits_C`` mapping.
    """
    repos = raw.get("repositories")
    if repos:
        return {k: float(v["surface_limit_C"]) for k, v in repos.items()}
    return {k: float(v) for k, v in raw["surface_limits_C"].items()}


def load_config(
    yaml_path: Optional[str] = None,
    material: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and parse ``solver_config.yaml``.

    Parameters
    ----------
    yaml_path : str or Path, optional
        Path to config file. Defaults to ``solver_config.yaml`` in the current
        working directory.
    material : str, optional
        Override the material selected by the config's ``material`` key.

    Returns
    -------
    dict
        Structured configuration with keys:

        - ``waste_form_name`` (str) — label used in output filenames
        - ``waste_form`` (dict) — ``rho_base`` (float), ``decay``/``cp``/``k`` (callables)
        - ``centerline_limit_C``, ``safety_factor`` (float)
        - ``surface_limits_C`` (dict) — geology → surface limit [°C]
        - ``passive_ambient_C``, ``passive_h`` (float) — interim store / repository
        - ``h_passive`` (float) — alias of ``passive_h``
        - ``cooling_archetypes`` (dict) — the technology library
        - ``geology_names``, ``archetype_names`` (list of str or None) — which
          to compare; ``None`` means every one defined
        - ``total_waste_mass_kg`` (float or None), ``canister_aspect_ratio`` (float)
        - ``pre_encap_min_years``, ``pre_encap_max_years`` (float)
        - ``radii_min``, ``radii_max``, ``radii_steps`` (float / int)
        - ``loadings_pct`` (list of float or None) — explicit override
        - ``loadings_min``, ``loadings_max``, ``loadings_steps`` (float / int)
        - ``candidates`` (list of dict) — named designs to report, may be empty
        - ``nodes``, ``max_years`` (int / float)
    """
    if yaml_path is None:
        yaml_path = Path.cwd() / "solver_config.yaml"
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        # Almost always caused by running from outside the repository, so say
        # that rather than leaving a bare "no such file" to be interpreted.
        raise FileNotFoundError(
            f"No configuration file at {yaml_path}\n"
            "  solver_config.yaml is read from the directory you run in.\n"
            "  Either cd into the repository, or point at it directly:\n"
            "      aethon --config /path/to/solver_config.yaml"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if material is not None:
        raw["material"] = material

    # --- Waste stream: preprocessor file wins, then inline ------------------
    source = _resolve_waste_source(raw, yaml_path.parent)
    decay_terms = source.get("decay_terms") or raw.get("decay_terms")

    # Where the decay curve came from, so callers can report it. Inline terms
    # are frequently the shipped placeholders, and a user who meant to point at
    # a preprocessor output has no other way to notice.
    waste_source_path = str(source["_path"]) if source.get("_path") else None

    # --- Material -----------------------------------------------------------
    (waste_form, material_name, material_limit,
     material_spec, resolved_terms) = _resolve_material(raw, decay_terms)

    # centerline_limit_C is a material property; top level is the fallback
    if material_limit is None:
        material_limit = raw["centerline_limit_C"]

    # --- Phase conditions ---------------------------------------------------
    passive = raw.get("passive", {})
    campaign = raw.get("campaign", {})
    pre_encap = raw.get("pre_encapsulation_years", {})

    # Active-facility ambient comes from the chosen cooling archetype, never
    # from a global setting — an HTC and the temperature it works against are
    # a matched pair. Only the passive phases have a site-wide ambient.
    passive_ambient_C = float(
        passive.get("ambient_C", raw.get("ambient_temp_C", 50.0))
    )
    passive_h = float(passive.get("h", raw.get("h_passive", 5.0)))

    # Campaign size is an operational fact, not a property of the decay curve,
    # so it has exactly one home: the config. The preprocessor does not set it.
    total_mass = campaign.get("total_waste_mass_kg")

    # Every range in the file has the same shape - a block with min, max and
    # (where it is a grid) steps. Flattened here because the solver reads them
    # as separate scalars.
    radii = _require_range(raw, "radii", steps=True)
    loadings = _require_range(raw, "loadings", steps=True)

    return {
        "waste_form_name": material_name,
        "waste_form": waste_form,
        "waste_source_path": waste_source_path,
        "material_spec": material_spec,
        "decay_terms": resolved_terms,
        "centerline_limit_C": float(material_limit),
        "safety_factor": float(raw["safety_factor"]),
        "surface_limits_C": _resolve_surface_limits(raw),
        "passive_ambient_C": passive_ambient_C,
        "passive_h": passive_h,
        "h_passive": passive_h,
        "cooling_archetypes": raw.get("cooling_archetypes") or {},
        # What to compare, kept separate from the libraries that define them so
        # narrowing a run never means deleting a definition. ``None`` means
        # everything defined.
        "geology_names": (
            [str(n) for n in raw["geologies"]]
            if raw.get("geologies") else None
        ),
        "archetype_names": (
            [str(n) for n in raw["archetypes"]]
            if raw.get("archetypes") else None
        ),
        "total_waste_mass_kg": None if total_mass is None else float(total_mass),
        "canister_aspect_ratio": float(campaign.get("canister_aspect_ratio", 6.0)),
        "pre_encap_min_years": float(pre_encap.get("min", 0.0)),
        "pre_encap_max_years": float(pre_encap.get("max", 10.0)),
        "radii_min": float(radii["min"]),
        "radii_max": float(radii["max"]),
        "radii_steps": int(radii["steps"]),
        "loadings_min": float(loadings["min"]),
        "loadings_max": float(loadings["max"]),
        "loadings_steps": int(loadings["steps"]),
        # Not offered in the config: an explicit grid cannot be expressed as a
        # range, so run records write one here to pin exactly what was swept.
        # Users reach it through --loadings instead.
        "loadings_pct": (
            [float(x) for x in raw["loadings_pct"]]
            if raw.get("loadings_pct") else None
        ),
        "candidates": raw.get("candidates") or [],
        "nodes": int(raw["nodes"]),
        "max_years": float(raw["max_years"]),
    }
