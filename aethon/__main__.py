"""
CLI entry point for AETHON (Analysis of Encapsulated Thermal Heat and Optimised Nuclides).

Sweeps canister radius against waste loading and maps the three storage
milestones over that plane, comparing every cooling technology and repository
geology side by side.  Designs named in the config's ``candidates`` block are
additionally reported exactly.

Usage examples::

    # Compare everything, using solver_config.yaml as-is
    aethon

    # A conventional LWR operator with years of pool storage available
    aethon --t-pre-min 5 --t-pre-max 10 --material BorosilicateGlass

    # An aggressive recycling scheme with weeks, not years
    aethon --t-pre-max 0.25 --repo Salt

    # What cooling technologies are available?
    aethon --list-archetypes

    # Also runnable as a module
    python -m aethon
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from aethon import console
from aethon.config_loader import load_config
from aethon.design.archetypes import resolve_archetypes
from aethon.design.candidates import evaluate_candidates, parse_candidates
from aethon.design.report import plot_design_maps, sweep_stats
from aethon.design.search import (
    resolve_loadings,
    resolve_selection,
    run_exploration,
)
from aethon.run_record import write_run_record


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.  Defaults come from the config file."""
    parser = argparse.ArgumentParser(
        prog="aethon",
        description=(
            "Sweep canister radius against waste loading and map when the "
            "waste can be encapsulated, when the coolers can stop, and when "
            "the repository will accept it. Cooling technologies and "
            "repository geologies are enumerated and compared rather than "
            "optimised over."
        ),
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to solver_config.yaml (default: ./solver_config.yaml).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for CSVs and plots (default: results/).",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plot generation.",
    )

    parser.add_argument(
        "--material", type=str, default=None,
        help="Waste form material from the config's materials library.",
    )
    parser.add_argument(
        "--repo", type=str, nargs="+", default=None,
        help="Repository geologies to evaluate (default: all defined).",
    )
    parser.add_argument(
        "--archetype", type=str, nargs="+", default=None,
        help="Cooling technologies to evaluate (default: all available).",
    )
    parser.add_argument(
        "--t-pre-min", type=float, default=None,
        help="Earliest the waste can be encapsulated [years from shutdown].",
    )
    parser.add_argument(
        "--t-pre-max", type=float, default=None,
        help="Latest acceptable encapsulation [years from shutdown].",
    )
    parser.add_argument(
        "--total-mass", type=float, default=None,
        help="Total campaign waste mass [kg]; overrides the config value.",
    )
    parser.add_argument(
        "--safety-factor", type=float, default=None,
        help=(
            "Divisor applied to all temperature limits; 1.25 gives 20 percent "
            "margin. Overrides the config value."
        ),
    )
    parser.add_argument(
        "--aspect-ratio", type=float, default=None,
        help="Canister height / radius; overrides the config value.",
    )

    parser.add_argument(
        "--loadings", type=float, nargs="+", default=None,
        help=(
            "Explicit waste loading percentages (e.g. 5 10 15 20); overrides "
            "the loading range. Fast, but too coarse to contour."
        ),
    )
    parser.add_argument("--loadings-min", type=float, default=None,
                        help="Minimum waste loading [wt%%].")
    parser.add_argument("--loadings-max", type=float, default=None,
                        help="Maximum waste loading [wt%%].")
    parser.add_argument("--loadings-steps", type=int, default=None,
                        help="Number of loading points to evaluate.")
    parser.add_argument("--radii-min", type=float, default=None,
                        help="Minimum canister radius [m].")
    parser.add_argument("--radii-max", type=float, default=None,
                        help="Maximum canister radius [m].")
    parser.add_argument("--radii-steps", type=int, default=None,
                        help="Number of radius points to evaluate.")

    parser.add_argument(
        "--list-archetypes", action="store_true",
        help="Print the available cooling technologies and exit.",
    )

    return parser


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Fold CLI overrides into the loaded configuration."""
    if args.t_pre_min is not None:
        cfg["pre_encap_min_years"] = args.t_pre_min
    if args.t_pre_max is not None:
        cfg["pre_encap_max_years"] = args.t_pre_max
    if args.total_mass is not None:
        cfg["total_waste_mass_kg"] = args.total_mass
    if args.safety_factor is not None:
        cfg["safety_factor"] = args.safety_factor
    if args.aspect_ratio is not None:
        cfg["canister_aspect_ratio"] = args.aspect_ratio

    if cfg["safety_factor"] <= 0.0:
        raise ValueError(
            f"safety_factor must be positive, got {cfg['safety_factor']:g}. "
            "It divides every temperature limit, so 1.0 means no margin."
        )

    if cfg["pre_encap_min_years"] > cfg["pre_encap_max_years"]:
        raise ValueError(
            f"Pre-encapsulation window is empty: min "
            f"({cfg['pre_encap_min_years']:g} yr) exceeds max "
            f"({cfg['pre_encap_max_years']:g} yr)."
        )
    return cfg


def _resolve_radii(cfg: dict, args: argparse.Namespace) -> np.ndarray:
    """
    Radius grid from CLI overrides, falling back to config.

    Log-spaced: the physics changes fastest at small radii, where the
    ``Q·R²/(4k)`` conduction term is still small.
    """
    r_min = args.radii_min if args.radii_min is not None else cfg["radii_min"]
    r_max = args.radii_max if args.radii_max is not None else cfg["radii_max"]
    steps = args.radii_steps if args.radii_steps is not None else cfg["radii_steps"]
    return np.geomspace(r_min, r_max, steps)


def _resolve_loadings(cfg: dict, args: argparse.Namespace) -> list:
    """
    Loading grid from CLI overrides, falling back to config.

    An explicit ``--loadings`` list wins outright; otherwise the range bounds
    are folded into the config and the shared resolver builds the grid, so the
    CLI and the config file cannot drift apart in how they interpret them.
    """
    if args.loadings:
        return [float(x) for x in args.loadings]

    cfg = dict(cfg)
    if args.loadings_min is not None:
        cfg["loadings_min"] = args.loadings_min
    if args.loadings_max is not None:
        cfg["loadings_max"] = args.loadings_max
    if args.loadings_steps is not None:
        cfg["loadings_steps"] = args.loadings_steps

    # A range flag is an instruction to sweep a range, so it overrides an
    # explicit list left in the config rather than being silently ignored.
    if any(v is not None for v in
           (args.loadings_min, args.loadings_max, args.loadings_steps)):
        cfg["loadings_pct"] = None

    return resolve_loadings(cfg)


def _print_archetypes(cfg: dict) -> None:
    """List the available cooling technologies and their design-basis conditions."""
    rows = [
        [name, f"{spec['h']:g}", f"{spec['ambient_C']:g}",
         spec.get("description", "")]
        for name, spec in resolve_archetypes(cfg).items()
    ]
    console.data_table(
        ["Technology", "h [W/(m2.K)]", "Ambient [degC]", "Description"],
        rows, title="Available cooling technologies",
        align=["left", "right", "right", "left"],
    )
    console.hint("Literature-typical convective ranges for orientation, not "
                 "vendor data.")
    console.hint("Edit them in the 'cooling_archetypes' block of "
                 "solver_config.yaml.")


def _print_plan(radii, loadings, repos: dict, archetypes: dict) -> None:
    """
    What the run is about to do, before it starts doing it.

    The encapsulation gate is one transient solve per grid point per cooling
    technology and can take minutes, so the count is worth seeing while there
    is still time to interrupt and coarsen the grid.
    """
    n_designs = len(radii) * len(loadings)
    console.key_values([
        ("Grid", f"{len(radii)} radii x {len(loadings)} loadings "
                 f"= {n_designs} designs"),
        ("Geologies", ", ".join(repos) or "none"),
        ("Technologies", ", ".join(archetypes) or "none"),
        ("Transient solves", f"{n_designs * len(archetypes)}"),
    ], title="Plan")


def run(args: argparse.Namespace) -> int:
    """Execute a design-space sweep."""
    cfg = load_config(args.config, material=args.material)

    if args.list_archetypes:
        _print_archetypes(cfg)
        return 0

    cfg = _apply_overrides(cfg, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    material = cfg["waste_form_name"]
    radii = _resolve_radii(cfg, args)
    loadings = _resolve_loadings(cfg, args)

    # Parsed before the sweep so a malformed candidate fails in a second
    # rather than after several minutes of transient solves.
    candidates = parse_candidates(cfg)
    repos, archetypes = resolve_selection(cfg, args.repo, args.archetype)

    console.blank()
    console.print_provenance(cfg, material)
    _print_plan(radii, loadings, repos, archetypes)

    full_df = run_exploration(
        cfg=cfg,
        repositories=args.repo,
        archetype_names=args.archetype,
        radii=radii,
        loadings_pct=loadings,
    )

    full_path = output_dir / f"explore_full_{material}.csv"
    full_df.to_csv(full_path, index=False)

    record_path = write_run_record(
        output_dir=output_dir,
        cfg=cfg,
        repositories=repos,
        archetypes=archetypes,
        radii=radii,
        loadings_pct=loadings,
    )

    console.blank()
    console.rule("Results")
    console.print_sweep_summary(sweep_stats(full_df))

    written = [(full_path, "every design evaluated, one row per combination")]

    if candidates:
        candidates_df = evaluate_candidates(cfg, repos, archetypes, candidates)
        console.blank()
        console.print_candidates(candidates_df)
        candidates_path = output_dir / f"candidates_{material}.csv"
        candidates_df.to_csv(candidates_path, index=False)
        written.append((candidates_path, "the named designs, in full"))

    console.blank()
    console.print_milestone_glossary()

    if not args.no_plot:
        for path in plot_design_maps(full_df, output_dir, material):
            label = ("when the coolers can stop, and when each repository "
                     "will accept" if "passive" in path.name
                     else "earliest encapsulation, per cooling technology")
            written.append((path, label))

    written.append((record_path, "the settings that produced all of this"))

    console.blank()
    console.rule("Written")
    console.file_list(written)
    console.blank()
    return 0


def main(argv=None) -> int:
    """Parse arguments and run."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        console.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
