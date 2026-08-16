"""
Active cooling technology archetypes.

An engineer designing a waste facility does not know, and cannot sensibly be
asked for, the ambient air temperature of a cooling hall that has not been
built yet — and in operation that temperature drifts with season, load, and
equipment condition.  Asking for a single number invites a guess that silently
determines the answer.

So the tool does not ask.  Instead the user picks a *cooling technology*, and
each technology carries a worst-case design-basis pair of ``(h, ambient_C)``
taken from standard convective heat-transfer ranges.  Choosing "forced air"
rather than "h = 25 and 40 °C" is a question an engineer can actually answer,
and the worst-case framing means real operation can only be better than the
prediction.

Only the product of the pair matters physically — convective flux is
``h·(T_surface − T_ambient)`` — so a cooler hall and a higher HTC are
interchangeable ways of buying the same margin.  Bundling them removes a
degree of freedom the user would otherwise have to reason about.

.. warning::
   These are literature-typical convective ranges for orientation, **not**
   vendor performance data.  Replace them with values from your own facility
   design before using results for anything that matters — they are written
   out in the ``cooling_archetypes`` block of ``solver_config.yaml``, so edit
   them there.

The dict below duplicates that block.  It is the fallback for a configuration
that omits it, so the package stays usable from Python without a config file;
``tests/test_archetypes.py`` pins the two against each other.
"""

from typing import Any, Dict


# Ordered weakest → strongest so reports read naturally and the cheapest
# viable technology appears first.
BUILTIN_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "NaturalAir": {
        "h": 5.0,
        "ambient_C": 40.0,
        "description": "Unforced air in a passive vault; no cooling plant.",
    },
    "ForcedAir": {
        "h": 25.0,
        "ambient_C": 40.0,
        "description": "Fan-driven air over the canister surface.",
    },
    "WaterPool": {
        "h": 750.0,
        "ambient_C": 40.0,
        "description": "Full immersion in a cooled water pool.",
    },
}

_REQUIRED_KEYS = ("h", "ambient_C")


def resolve_archetypes(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    The cooling technology library for this run.

    ``cooling_archetypes`` **defines** the library outright — the same way
    ``materials`` and ``repositories`` do.  What you write there is what
    exists; deleting an entry removes that technology.  The block is shipped
    filled in, so correcting a figure means editing it in place rather than
    restating anything.

    To compare a subset without deleting definitions, use the ``archetypes``
    selection list (or ``--archetype``); see :func:`select_archetypes`.

    The dict above is the fallback for a configuration that omits the block
    entirely, which keeps the package usable from Python without a config
    file.  It is not a base that user entries are layered onto: an earlier
    version merged the two, and a user who listed one technology reasonably
    expected the others to disappear and got a run comparing all of them.

    Parameters
    ----------
    cfg : dict
        Parsed configuration; reads the ``cooling_archetypes`` key.

    Returns
    -------
    dict
        Archetype name → ``{"h", "ambient_C", "description"}``.

    Raises
    ------
    ValueError
        If an archetype is missing ``h`` or ``ambient_C``.
    """
    defined = cfg.get("cooling_archetypes") or {}
    if not defined:
        return {name: dict(spec) for name, spec in BUILTIN_ARCHETYPES.items()}

    library: Dict[str, Dict[str, Any]] = {}
    for name, spec in defined.items():
        missing = [key for key in _REQUIRED_KEYS if key not in (spec or {})]
        if missing:
            raise ValueError(
                f"Cooling archetype '{name}' is missing required "
                f"key(s): {', '.join(missing)}. Each archetype needs "
                "'h' [W/(m2.K)] and 'ambient_C' [degC]."
            )
        entry = dict(spec)
        entry.setdefault("description", "User-defined cooling technology.")
        library[name] = entry

    return library


def select_archetypes(
    cfg: Dict[str, Any], names: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Resolve the archetype library and narrow it to the selected names.

    Selection is a separate concern from definition, and needs its own key.
    ``cooling_archetypes`` *merges into* the built-in library rather than
    replacing it — which is what lets a user correct one technology's figures
    without restating the others — so it can never express "ignore the rest".
    The ``archetypes`` list does that instead.

    Precedence is the same as every other setting: an explicit *names*
    argument, normally from ``--archetype``, overrides the config.

    Parameters
    ----------
    cfg : dict
        Parsed configuration.  Read for ``archetype_names`` when *names* is
        not given.
    names : list of str, optional
        Archetype names to keep.  ``None`` falls back to the config, and then
        to every technology defined.

    Returns
    -------
    dict
        The selected subset, in the order it was requested.

    Raises
    ------
    ValueError
        If a requested name is not defined.
    """
    library = resolve_archetypes(cfg)

    if not names:
        names = cfg.get("archetype_names")
        source = "the config's 'archetypes' list"
    else:
        source = "--archetype"

    if not names:
        return library

    unknown = [n for n in names if n not in library]
    if unknown:
        available = ", ".join(library)
        raise ValueError(
            f"Unknown cooling archetype(s) in {source}: "
            f"{', '.join(unknown)}. Available: {available}"
        )
    return {n: library[n] for n in names}
