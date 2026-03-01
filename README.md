# Thermal Envelope Solver

A tool for computing the **cooling schedule** of vitrified nuclear waste canisters.
Given a canister geometry and waste loading, it produces:

1. **Minimum active HTC** — the convective heat transfer coefficient required to keep
   the glass below its transition temperature during peak decay heat.
2. **Minimum cooling time** — years of active cooling needed before the canister can
   safely transition to passive repository storage.

Results are swept over a range of canister radii and waste loading fractions and
output as a CSV and stacked design-envelope plot.

See [**QUICKSTART.md**](QUICKSTART.md) for the 3-step setup guide.

---

## Installation

```bash
pip install -r requirements.txt
```

No compiled dependencies. Python 3.10+ recommended.

---

## Quick start

```bash
# Run with the example configuration
python main.py

# Specify repository geology
python main.py --repo Salt

# Override sweep resolution
python main.py --repo Bentonite --radii-steps 200 --loadings 5 10 20

# Suppress plot output
python main.py --no-plot
```

Output is written to `results/` by default.

---

## File layout

| Path | Who edits it |
|------|-------------|
| `solver_config.yaml` | **You** — all thermal solver settings |
| `decay_preprocessor/preprocessor_config.yaml` | **You** — only if using the decay preprocessor |
| `examples/` | Reference input files |
| `data/` | Bundled decay chain data (do not edit) |
| `src/` | Solver internals (do not edit) |
| `decay_preprocessor/` | Preprocessor internals (do not edit) |
| `tests/` | Test suite (do not edit) |

---

## Configuration

All parameters are set in `solver_config.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `waste_form_name` | Label used in output filenames |
| `waste_form` | Material properties: density, decay heat, cp(T), k(T) |
| `centerline_limit_C` | Glass-transition temperature limit [°C] |
| `safety_factor` | Divisor applied to all temperature limits |
| `surface_limits_C` | Repository surface temperature limits by geology |
| `radii_min/max/steps` | Canister radius sweep range and resolution |
| `loadings_pct` | Waste loading percentages to evaluate |
| `max_h_active` | Feasibility ceiling for active HTC [W/(m²·K)] |
| `max_cooling_years` | Feasibility ceiling for cooling time [years] |

### Decay heat parameters

The `decay_terms` field under `waste_form` defines the specific decay heat curve:

```
Q(t) = Σ Aᵢ · exp(−λᵢ · t)   [W/kg],   t in years
```

If you have an isotope inventory, use the **decay preprocessor** to compute these
parameters automatically.

---

## Decay preprocessor

The `decay_preprocessor/` tool converts a full isotope inventory into fitted
decay heat parameters. Run it **once** before using the thermal solver.

Edit `decay_preprocessor/preprocessor_config.yaml` to set your inventory path,
sample mass, and chain file, then run:

```bash
# Zero-flag run (uses preprocessor_config.yaml entirely):
python -m decay_preprocessor.run_preprocessor

# Auto-write fitted terms directly into solver_config.yaml:
python -m decay_preprocessor.run_preprocessor --update-config
```

See [`decay_preprocessor/README.md`](decay_preprocessor/README.md) for full
documentation including the chain XML format and where to obtain chain files.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Physics

The solver uses:

- **Transient active phase**: 1D cylindrical Method of Lines FD solver
  (`scipy.integrate.solve_ivp`, BDF). Terminates at peak centreline temperature.
- **Passive phase safety check**: Analytical steady-state cylindrical solution
  inverted against repository surface and centreline limits.
- **Root-finding**: `scipy.optimize.brentq` for both minimum-h and minimum-cooling-time searches.

The governing PDE is:

```
ρ·Cₚ(T)·∂T/∂t = (1/r)·∂/∂r[r·k(T)·∂T/∂r] + Q_vol(t)
```

with a Robin (convection) boundary condition at the canister surface and
symmetry at the centreline.
