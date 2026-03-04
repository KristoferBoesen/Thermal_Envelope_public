"""
CLI entry point for the decay heat preprocessor.

Reads an isotope inventory CSV, runs the full Bateman decay chain solver,
fits a sum-of-exponentials to the output, and prints the fitted parameters
formatted for direct paste into ``solver_config.yaml``.

Usage::

    decay-preprocessor \\
        --inventory path/to/inventory.csv \\
        --chain     path/to/chain.xml \\
        --sample-mass 100.0

    # Auto-write fitted terms back into solver_config.yaml:
    decay-preprocessor --inventory ... --chain ... --sample-mass ... --update-config

The inventory CSV must have two columns:

- ``Isotope`` — nuclide name string matching the chain file (e.g. ``"Co60"``)
- ``Atoms``   — number of atoms at t = 0

Chain XML files (OpenMC format) can be obtained from:
    https://openmc.org/nuclear-data/

Output files written to ``--output-dir`` (default: current directory):

- ``decay_curve.csv`` — raw Bateman solution (Time_Years, Heat_Watts, Specific_Power_W_kg)
- ``decay_fit.png``   — diagnostic plot comparing Bateman solution to fitted curve
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from decay_preprocessor.chain_parser import parse_chain
from decay_preprocessor.bateman_solver import solve_decay
from decay_preprocessor.decay_fitter import fit_decay_curve, plot_fit


def _write_decay_terms_to_config(terms, config_path):
    """
    Replace only the ``decay_terms:`` block in *config_path*, preserving all
    other YAML content and comments.

    Parameters
    ----------
    terms : list of (float, float)
        Fitted ``(Amplitude, DecayConstant)`` pairs.
    config_path : Path
        Absolute path to the root ``solver_config.yaml`` to update.
    """
    lines = Path(config_path).read_text(encoding="utf-8").splitlines(keepends=True)

    new_block = ["  decay_terms:\n"]
    for A, lam in terms:
        new_block.append(f"    - [{A:.6g}, {lam:.6g}]\n")

    out = []
    old_terms_display = []
    in_decay_terms = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("decay_terms:"):
            in_decay_terms = True
            out.extend(new_block)
            continue
        if in_decay_terms:
            if stripped.startswith("- ["):
                old_terms_display.append(stripped)
                continue
            else:
                in_decay_terms = False
        out.append(line)

    print("\nUpdating solver_config.yaml decay_terms:")
    print("  Before:", old_terms_display or ["(none)"])
    print("  After: ", [f"[{A:.6g}, {lam:.6g}]" for A, lam in terms])
    Path(config_path).write_text("".join(out), encoding="utf-8")
    print(f"  -> Saved: {config_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Decay Heat Preprocessor — Bateman solver + exponential fitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Required arguments --------------------------------------------------
    parser.add_argument(
        "--inventory", required=True,
        help=(
            "Path to isotope inventory CSV (columns: Isotope, Atoms). "
            "Lines starting with '#' are treated as comments."
        ),
    )
    parser.add_argument(
        "--chain", required=True,
        help=(
            "Path to OpenMC-format decay chain XML file. "
            "Chain files can be obtained from https://openmc.org/nuclear-data/"
        ),
    )
    parser.add_argument(
        "--sample-mass", type=float, required=True,
        help=(
            "Mass of the WASTE MATERIAL ONLY [kg]. "
            "Do not use total canister or composite mass."
        ),
    )

    # --- Optional arguments --------------------------------------------------
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Simulation duration [years] (default: 10.0).",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for output files (default: current directory).",
    )
    parser.add_argument(
        "--n-points", type=int, default=2000,
        help="Number of log-spaced time evaluation points (default: 2000).",
    )
    parser.add_argument(
        "--cutoff-years", type=float, default=1 / 12,
        help=(
            "Exclude data before this time [yr] when fitting (default: 1/12 ≈ 1 month). "
            "Discards short-lived nuclides irrelevant to long-term cooling."
        ),
    )
    parser.add_argument(
        "--update-config", action="store_true",
        help=(
            "Write fitted decay_terms back into solver_config.yaml in the current "
            "directory. Only the decay_terms block is changed; all other content "
            "is preserved."
        ),
    )

    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Parse chain file --------------------------------------------
    print(f"[1/4] Parsing chain file:  {args.chain}")
    nuc_to_idx, decay_constants, q_values, matrix_A = parse_chain(args.chain)
    print(f"      {len(nuc_to_idx):,} nuclides loaded.")

    # --- Step 2: Load inventory ----------------------------------------------
    print(f"[2/4] Loading inventory:   {args.inventory}")
    inventory_df = pd.read_csv(args.inventory, comment="#")
    n_matched = sum(1 for iso in inventory_df["Isotope"] if iso in nuc_to_idx)
    print(
        f"      {len(inventory_df)} isotopes in inventory, "
        f"{n_matched} matched to chain."
    )
    if n_matched == 0:
        print(
            "ERROR: No inventory isotopes matched the chain file. "
            "Check nuclide naming conventions.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Step 3: Solve Bateman equations -------------------------------------
    print(f"[3/4] Solving decay chain for {args.duration:.1f} years ...")
    result_df = solve_decay(
        inventory_df=inventory_df,
        nuc_to_idx=nuc_to_idx,
        decay_constants=decay_constants,
        q_values=q_values,
        matrix_A=matrix_A,
        sample_mass_kg=args.sample_mass,
        duration_years=args.duration,
        n_points=args.n_points,
    )
    csv_path = output_dir / "decay_curve.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"      Saved: {csv_path}")

    # --- Step 4: Fit sum-of-exponentials -------------------------------------
    print("[4/4] Fitting sum-of-exponentials ...")
    terms, r2, rmse = fit_decay_curve(
        result_df["Time_Years"].values,
        result_df["Specific_Power_W_kg"].values,
        cutoff_years=args.cutoff_years,
    )

    # --- Print YAML-ready output ---------------------------------------------
    print("\n" + "=" * 60)
    print(f"Fit quality:  R² = {r2:.8f}  |  RMSE = {rmse:.2f} W/kg")
    print(f"Terms fitted: {len(terms)}")
    print("=" * 60)
    print("\nPaste the following into solver_config.yaml under 'waste_form':\n")
    print("  decay_terms:")
    for A, lam in terms:
        print(f"    - [{A:.6g}, {lam:.6g}]")
    print()

    # --- Save diagnostic plot ------------------------------------------------
    plot_path = output_dir / "decay_fit.png"
    plot_fit(
        result_df["Time_Years"].values,
        result_df["Specific_Power_W_kg"].values,
        terms,
        r2,
        output_path=plot_path,
    )
    print(f"Plot saved:   {plot_path}")

    # --- Optionally write fitted terms back to solver_config.yaml -----------
    if args.update_config:
        root_config = Path.cwd() / "solver_config.yaml"
        try:
            _write_decay_terms_to_config(terms, root_config)
        except Exception as exc:
            print(f"  Warning: could not write solver_config.yaml: {exc}")
            print("  Paste the terms above manually.")

    print("\nDone.")


if __name__ == "__main__":
    main()
