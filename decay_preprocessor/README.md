# Decay Heat Preprocessor

A standalone preprocessing tool that converts a full isotope inventory into
fitted decay heat parameters for use in `config.yaml`.

Run this tool **once** before using the thermal envelope solver whenever you
have a new isotope inventory. The thermal solver itself only needs the fitted
exponential coefficients.

---

## What it does

1. **Parses** an OpenMC-format decay chain XML file (no OpenMC installation required)
2. **Solves** the full Bateman decay chain equations using a stiff BDF integrator
3. **Fits** a sum-of-exponentials `Q(t) = Σ Aᵢ · exp(−λᵢ · t)` to the result
4. **Outputs** the fitted parameters formatted for direct paste into `config.yaml`,
   along with a diagnostic plot and R² quality metric

---

## Usage

```bash
python -m decay_preprocessor.run_preprocessor \
    --inventory path/to/inventory.csv \
    --chain path/to/chain.xml \
    --sample-mass 115.98 \
    --duration 10.0 \
    --output-dir results/decay/
```

### Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--inventory` | Yes | Path to isotope inventory CSV |
| `--chain` | Yes | Path to decay chain XML file |
| `--sample-mass` | Yes | Total sample mass [kg] |
| `--duration` | No | Simulation duration in years (default: 10.0) |
| `--n-terms` | No | Fixed number of exponential terms (default: auto 3–6) |
| `--output-dir` | No | Output directory (default: current directory) |

---

## Input formats

### Isotope inventory CSV

Two columns (lines starting with `#` are ignored):

```csv
# My waste form inventory at t=0
Isotope,Atoms
Cs137,1.20e+20
Sr90,8.50e+19
Co60,2.30e+18
```

Nuclide names must match the naming convention used in your chain XML file.

### Decay chain XML

Standard OpenMC chain format. Each nuclide:

```xml
<chain>
  <nuclide name="Co60" half_life="1.6625e+08">
    <decay type="beta-" target="Ni60" branching_ratio="1.0" energy="96624.64"/>
  </nuclide>
  <nuclide name="Ni60"/>  <!-- stable: no half_life -->
</chain>
```

The standard OpenMC chain files (e.g. `chain_endfb80_pwr.xml`) can be
downloaded from the [OpenMC data repository](https://openmc.org/nuclear-data/).

---

## Output

- **Console**: fitted parameters formatted as YAML, ready to paste into `config.yaml`
- **`decay_curve.csv`**: full Bateman solution (Time_Years, Heat_Watts, Specific_Power_W_kg)
- **`decay_fit.png`**: diagnostic plot comparing the Bateman solution to the fitted curve

### Example console output

```
[1/4] Parsing chain file:  chain_endfb80_pwr.xml
      1681 nuclides loaded.
[2/4] Loading inventory:   inventory.csv
      45 isotopes in inventory, 43 matched to chain.
[3/4] Solving decay chain for 10.0 years ...
      Saved: results/decay/decay_curve.csv
[4/4] Fitting sum-of-exponentials ...

============================================================
Fit quality:  R² = 0.99998421
Terms fitted: 4
============================================================

Paste the following into config.yaml under 'waste_form':

  decay_terms:
    - [982.77, 24.18]
    - [733.59,  3.88]
    - [ 44.57,  0.02]
    - [  1.23,  0.001]

Plot saved:   results/decay/decay_fit.png
```
