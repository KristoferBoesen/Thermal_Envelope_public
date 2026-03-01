"""
CLI entry point for the Thermal Envelope Design Tool.

Usage examples::

    # Run with defaults from solver_config.yaml
    python main.py

    # Specify repository geology
    python main.py --repo Salt

    # Override sweep parameters
    python main.py --repo Bentonite --loadings 5 10 --radii-steps 50

    # Suppress plot generation
    python main.py --no-plot
"""

import argparse
import numpy as np
from pathlib import Path

from thermal_envelope.config_loader import load_config
from thermal_envelope.analysis.pipeline import run_design_envelope
from thermal_envelope.analysis.plotting import plot_design_envelope


def parse_args(cfg: dict, argv=None) -> argparse.Namespace:
    """Build argument parser with defaults drawn from solver_config.yaml."""
    repo_choices = list(cfg["surface_limits_C"].keys())

    parser = argparse.ArgumentParser(
        description="Nuclear Waste Canister — Thermal Design Envelope Generator",
    )
    parser.add_argument(
        "--repo", type=str, default="Bentonite",
        choices=repo_choices,
        help="Repository geology type (default: Bentonite).",
    )
    parser.add_argument(
        "--loadings", type=float, nargs="+",
        default=cfg["loadings_pct"],
        help="Waste loading percentages (e.g. 5 10 15 20).",
    )
    parser.add_argument(
        "--radii-min", type=float, default=cfg["radii_min"],
    )
    parser.add_argument(
        "--radii-max", type=float, default=cfg["radii_max"],
    )
    parser.add_argument(
        "--radii-steps", type=int, default=cfg["radii_steps"],
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for CSV and plots (default: results/).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    cfg = load_config()
    args = parse_args(cfg, argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = cfg["waste_form_name"]
    props = cfg["waste_form"]
    radii = np.linspace(args.radii_min, args.radii_max, args.radii_steps)

    print(f"[{label} / {args.repo}] Computing design envelope...")

    df = run_design_envelope(
        label=label,
        properties=props,
        repo_type=args.repo,
        loadings_pct=args.loadings,
        radii=radii,
        cfg=cfg,
    )

    csv_name = f"Design_Envelope_{label}_{args.repo}.csv"
    csv_path = output_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"  -> Saved: {csv_path}")

    if not args.no_plot:
        fig_path = plot_design_envelope(
            df, label, args.repo, output_dir,
            max_h=cfg.get("max_h_active"),
            max_cool=cfg.get("max_cooling_years"),
        )
        print(f"  -> Plot:  {fig_path}")

    print("Done.")


if __name__ == "__main__":
    main()
