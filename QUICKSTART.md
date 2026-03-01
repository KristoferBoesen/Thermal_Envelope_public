# Quick Start

## Step 1 — Install

```bash
pip install -r requirements.txt
```

Or, for a proper install that adds `thermal-envelope` as a shell command:

```bash
pip install -e .
```

---

## Step 2 — Configure

Edit **`solver_config.yaml`** (project root). Key fields:

| Field | Description |
|-------|-------------|
| `waste_form_name` | Label used in output filenames |
| `centerline_limit_C` | Glass-transition temperature limit [°C] |
| `safety_factor` | Divisor applied to all temperature limits (≥ 1.0) |
| `waste_form.decay_terms` | Fitted decay heat curve — see Step 2b |
| `waste_form.cp_poly` / `k_poly` | Specific heat and conductivity as polynomial coefficients |
| `radii_min` / `radii_max` / `radii_steps` | Canister radius sweep range and resolution |
| `loadings_pct` | Waste loading percentages to evaluate |

### Step 2b — Generate decay terms from an isotope inventory (optional)

If you have an isotope inventory CSV, edit
**`decay_preprocessor/preprocessor_config.yaml`**:

- Set `inventory:` to your CSV path (columns: `Isotope`, `Atoms`)
- Set `sample_mass_kg:` to the mass of the waste material in kg
- Set `chain:` to the OpenMC decay chain XML path (default: `data/chain_endfb71_pwr.xml`)

Then run:

```bash
python -m decay_preprocessor.run_preprocessor --update-config
```

This runs the Bateman chain solver, fits the result to a sum-of-exponentials,
and writes `decay_terms` directly into `solver_config.yaml`. All other config
content is preserved.

See [`decay_preprocessor/README.md`](decay_preprocessor/README.md) for full
documentation.

---

## Step 3 — Run

```bash
python main.py --repo Bentonite
```

Or, if installed with `pip install -e .`:

```bash
thermal-envelope --repo Bentonite
```

Or on Windows, double-click (or run from a terminal):

```bat
run.bat --repo Bentonite
```

Results (CSV + plot) appear in `results/`.

---

## Common options

| Flag | Description |
|------|-------------|
| `--repo Bentonite` / `--repo Salt` | Repository geology type |
| `--loadings 5 10 20` | Waste loading percentages to evaluate |
| `--radii-steps 50` | Number of canister radii to sweep |
| `--no-plot` | Skip plot generation (faster for scripting) |
| `--output-dir path/` | Write results to a custom directory |

Full option list: `python main.py --help`
