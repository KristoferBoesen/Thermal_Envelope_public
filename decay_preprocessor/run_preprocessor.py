"""
CLI entry point for the decay heat preprocessor.

Reads an isotope inventory CSV, runs the full Bateman decay chain solver,
fits a sum-of-exponentials to the output, and prints the fitted parameters
formatted for direct paste into ``config.yaml``.

Usage::

    python -m decay_preprocessor.run_preprocessor \\
        --inventory path/to/inventory.csv \\
        --chain path/to/chain.xml \\
        --sample-mass 100.0 \\
        --duration 10.0

The inventory CSV must have two columns:

- ``Isotope`` — nuclide name string matching the chain file (e.g. ``"Co60"``)
- ``Atoms``   — number of atoms at t = 0

Output files written to ``--output-dir``:

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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Decay Heat Preprocessor — Bateman solver + exponential fitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--inventory", required=True,
        help="Path to isotope inventory CSV (columns: Isotope, Atoms)",
    )
    parser.add_argument(
        "--chain", required=True,
        help="Path to decay chain XML file",
    )
    parser.add_argument(
        "--sample-mass", type=float, required=True,
        help="Total sample mass [kg]",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Simulation duration [years] (default: 10.0)",
    )
    parser.add_argument(
        "--n-terms", type=int, default=None,
        help="Number of exponential terms (default: auto-select 3–6)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory for output files (default: current directory)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Parse chain file ---
    print(f"[1/4] Parsing chain file:  {args.chain}")
    nuc_to_idx, decay_constants, q_values, matrix_A = parse_chain(args.chain)
    print(f"      {len(nuc_to_idx):,} nuclides loaded.")

    # --- Step 2: Load inventory ---
    print(f"[2/4] Loading inventory:   {args.inventory}")
    inventory_df = pd.read_csv(args.inventory, comment="#")
    n_matched = sum(1 for iso in inventory_df["Isotope"] if iso in nuc_to_idx)
    print(
        f"      {len(inventory_df)} isotopes in inventory, "
        f"{n_matched} matched to chain."
    )
    if n_matched == 0:
        print(
            "ERROR: No inventory isotopes matched the chain file.  "
            "Check nuclide naming conventions.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Step 3: Solve Bateman equations ---
    print(f"[3/4] Solving decay chain for {args.duration:.1f} years ...")
    result_df = solve_decay(
        inventory_df=inventory_df,
        nuc_to_idx=nuc_to_idx,
        decay_constants=decay_constants,
        q_values=q_values,
        matrix_A=matrix_A,
        sample_mass_kg=args.sample_mass,
        duration_years=args.duration,
    )
    csv_path = output_dir / "decay_curve.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"      Saved: {csv_path}")

    # --- Step 4: Fit sum-of-exponentials ---
    print("[4/4] Fitting sum-of-exponentials ...")
    terms, r2 = fit_decay_curve(
        result_df["Time_Years"].values,
        result_df["Specific_Power_W_kg"].values,
        n_terms=args.n_terms,
    )

    # --- Print YAML-ready output ---
    print("\n" + "=" * 60)
    print(f"Fit quality:  R² = {r2:.8f}")
    print(f"Terms fitted: {len(terms)}")
    print("=" * 60)
    print("\nPaste the following into config.yaml under 'waste_form':\n")
    print("  decay_terms:")
    for A, lam in terms:
        print(f"    - [{A:.6g}, {lam:.6g}]")
    print()

    # --- Save diagnostic plot ---
    plot_path = output_dir / "decay_fit.png"
    plot_fit(
        result_df["Time_Years"].values,
        result_df["Specific_Power_W_kg"].values,
        terms,
        r2,
        output_path=plot_path,
    )
    print(f"Plot saved:   {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
