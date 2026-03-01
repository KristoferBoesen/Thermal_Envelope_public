# Technical Overview — Thermal Envelope Solver

This document explains **what the solver does, why it does it that way, and how
all the pieces fit together.** It is written for someone who wants to understand
the tool at a level suitable for modifying, extending, or trusting its results —
without necessarily reading every source file.

---

## 1. The Physical Problem

### Nuclear waste canisters and decay heat

Vitrified high-level nuclear waste (HLW) is immobilised in borosilicate glass and
encased in a steel canister.  The waste glass contains fission products and
actinides that continue to undergo radioactive decay long after the reactor is
shut down, releasing heat in the process.

This **decay heat** is not negligible.  A freshly loaded canister can generate
tens to hundreds of watts per kilogram of waste glass.  If this heat cannot
escape fast enough, the centreline temperature of the glass rises to dangerous
levels.

### Why temperature matters

Borosilicate glass is stable (amorphous) up to roughly 500 °C.  Above this
**glass transition temperature** the structure begins to devitrify — crystalline
phases nucleate, the waste form loses integrity, and its long-term radionuclide
retention properties degrade.  Keeping the glass below this limit is a hard
safety requirement throughout the storage lifecycle.

### Two storage phases

Nuclear waste canisters pass through two distinct thermal environments:

1. **Active storage** — The canister sits in a ventilated vault or transport
   flask where forced convection provides controlled cooling.  The operator can
   dial up the heat transfer coefficient (HTC) as needed, but high-HTC systems
   are expensive and require maintenance.

2. **Passive repository** — The canister is eventually emplaced in a deep
   geological repository (DGR) where it relies on natural (passive) convection
   to the surrounding rock or engineered barrier.  The passive HTC is fixed and
   low (~5 W/(m²·K)).  The canister must be cool enough at the time of
   emplacement that it never exceeds the repository surface temperature limit
   (100 °C for Bentonite clay, 200 °C for salt rock) even under steady-state
   decay heating for all future time.

---

## 2. What the Solver Computes

For each combination of **canister radius** and **waste loading fraction**, the
solver finds:

### 2.1 Minimum active HTC

> "What is the smallest HTC that keeps the glass centreline below the safety
> limit during peak decay heat?"

This is the minimum forced-convection requirement during the active storage phase.
A result of, say, 45 W/(m²·K) means that normal forced-air ventilation is
sufficient; a result of 800 W/(m²·K) means only water cooling would work.

### 2.2 Minimum cooling time

> "How many years must the canister be actively cooled before it is safe to
> emplace in the repository under passive cooling alone?"

The decay heat falls over time.  Once it falls below the level that passive
cooling can safely handle (given the repository geometry and surface temperature
limit), active cooling can stop and the canister can be emplaced.

### The design envelope

By sweeping over a grid of radii and loadings, both results form a **design
envelope**: a 2D surface showing which combinations of geometry and waste loading
are feasible, and at what cost (HTC or waiting time).  This is the primary
output — a CSV and a stacked plot.

---

## 3. Solution Strategy

The solver splits the problem into two phases, each handled by a different
mathematical model.

### 3.1 Active phase — transient FEM solver

During active storage the decay power is high and changing rapidly.  The canister
temperature is not in steady state — it rises as the waste heats up and then
falls as decay heat decreases.  The **peak centreline temperature** (the worst
case) occurs somewhere in this transient.

The solver uses a **Method of Lines (MOL)** approach:

1. Discretise the 1D radial domain into `N` nodes using finite differences.
2. Write the heat equation at each node as an ODE in time.
3. Integrate the system of ODEs forward using `scipy.integrate.solve_ivp` with
   a BDF (backward differentiation formula) integrator -- appropriate because the
   system is stiff (wide range of timescales between nodes and between decay
   modes).
4. Terminate integration automatically when the centreline temperature reaches
   its peak (detected via a zero-crossing event on `dT_centre/dt`).

**Governing PDE** (cylindrical symmetry, no axial variation):

```
ρ·Cₚ(T)·∂T/∂t = (1/r)·∂/∂r[r·k(T)·∂T/∂r] + Q_vol(t)
```

where `Q_vol(t) = ρ_eff · Q_specific(t)` [W/m³], and `Q_specific(t)` [W/kg] is
the specific decay power defined by the fitted exponential terms.

**Boundary conditions:**

- **Centreline (r = 0):** Symmetry — no radial heat flux.  The `1/r` singularity
  is resolved via L'Hôpital's rule, which replaces the cylindrical term with a
  factor of 4 at the centre node.
- **Surface (r = R):** Robin (convective) — conduction out of the glass equals
  convection into the coolant: `−k · ∂T/∂r = h · (T_surface − T_∞)`.
  Implemented as a control-volume energy balance on the outermost half-cell.

**Material properties** are temperature-dependent: `k(T)` and `Cₚ(T)` are
quadratic polynomials in Kelvin, defined in `config.yaml` and evaluated at each
node at each time step.

### 3.2 Passive phase — analytical steady-state

Once peak active-phase behaviour is characterised, a separate question is: when
is the canister cool enough for passive storage?

This is a **steady-state** problem.  At any future time `t`, the decay power is
`Q(t)` [W/kg].  The worst-case temperatures (assuming the canister has been in
the repository forever at that power level) are:

```
T_centreline = T_∞ + Q·R/(2h_passive) + Q·R²/(4k)
T_surface    = T_∞ + Q·R/(2h_passive)
```

These are closed-form results for a cylinder with uniform internal heat generation,
convective outer boundary, and constant material properties.  The solver inverts
them: given the temperature limits, what is the maximum allowable `Q`?

```
Q_max_centre  = (T_limit_centre / SF  − T_∞) / [R/(2h) + R²/(4k)]
Q_max_surface = (T_limit_surface / SF − T_∞) / [R/(2h)]
Q_allowable   = min(Q_max_centre, Q_max_surface)
```

The **minimum cooling time** is then the earliest `t` at which `Q(t) ≤ Q_allowable`.
This is found by bisection (`scipy.optimize.brentq`) on the decay curve.

### 3.3 Root-finding loops

Finding the minimum HTC and minimum cooling time for each (radius, loading) point
involves two nested root-finding loops:

1. **Minimum HTC:** Binary search over `h` in [0.1, 2000] W/(m²·K).  For each
   candidate `h`, the full transient solver runs and returns the peak centreline
   temperature.  Brent's method drives this residual to zero.

2. **Minimum cooling time:** Binary search over `t` in [0, 1000] years.  At each
   candidate `t`, the analytical `Q(t)` is compared to `Q_allowable`.  Brent's
   method finds the crossing.

The double-loop structure means the transient solver runs many times per design
point.  The BDF integrator and event-based termination keep this tractable; log-
spaced grids ensure dense sampling at the physically interesting small-radius end.

---

## 4. Decay Heat Representation

The specific decay power `Q(t)` [W/kg] is represented throughout the solver as a
**sum of decaying exponentials**:

```
Q(t) = A₁·exp(−λ₁·t) + A₂·exp(−λ₂·t) + A₃·exp(−λ₃·t)
```

where `t` is in years, amplitudes `Aᵢ` are in W/kg, and decay constants `λᵢ` are
in yr⁻¹.  This form is:

- **Physically motivated** — each term approximates a group of nuclides with
  similar half-lives.
- **Analytically invertible** — the minimum cooling time calculation requires
  solving `Q(t) = Q_allowable`, which can be done with simple bisection.
- **Compact** — only 6 numbers in `config.yaml` represent the entire decay curve.

The parameters are determined by the **Decay Preprocessor** (see Section 6).

---

## 5. File Structure

```
Thermal_Envelope_public/
│
├── main.py                        # CLI entry point for the thermal solver
├── config.yaml                    # All user-adjustable parameters
├── requirements.txt
├── README.md                      # Quick-start reference
├── TECHNICAL_OVERVIEW.md          # This document
│
├── src/                           # Main thermal solver package
│   ├── config_loader.py           # Parses config.yaml → Python objects/callables
│   ├── constants.py               # Physical constants and solver bounds
│   ├── physics/
│   │   ├── fem_solver.py          # Transient 1D FD solver (WasteForm class)
│   │   └── analytical.py         # Steady-state analytical model
│   └── analysis/
│       ├── pipeline.py            # Sweep orchestration and root-finding loops
│       └── plotting.py            # Design envelope stacked plot
│
├── decay_preprocessor/            # Standalone isotope → decay curve tool
│   ├── chain_parser.py            # Parses OpenMC XML decay chain
│   ├── bateman_solver.py          # Solves Bateman equations (nuclide evolution)
│   ├── decay_fitter.py            # Fits 3-term exponential to Bateman output
│   ├── run_preprocessor.py        # CLI entry point for the preprocessor
│   └── README.md                  # Preprocessor-specific documentation
│
├── tests/                         # Automated test suite
│   ├── test_analytical.py         # Steady-state hand-calculation checks
│   ├── test_chain_parser.py       # XML parser for both depletion_chain/chain formats
│   ├── test_config.py             # Config loader and callable construction
│   └── test_fem_solver.py         # Transient solver physics checks
│
├── examples/
│   └── example_inventory.csv      # Sample isotope inventory for the preprocessor
│
└── chain_endfb71_pwr.xml          # OpenMC ENDF/B-VII.1 PWR decay chain (~1600 nuclides)
```

### Role of each major component

| File / module | Role |
|---|---|
| `config.yaml` | Single source of truth for all user inputs |
| `src/config_loader.py` | Bridges YAML → Python; builds `k(T)`, `Cₚ(T)`, `Q(t)` callables |
| `src/physics/fem_solver.py` | Solves the transient heat PDE; contains all FD discretisation |
| `src/physics/analytical.py` | Computes `Q_allowable` from geometry and limits |
| `src/analysis/pipeline.py` | Runs the two root-finding loops; produces the results DataFrame |
| `src/analysis/plotting.py` | Turns the DataFrame into the stacked design envelope plot |
| `decay_preprocessor/chain_parser.py` | Reads decay chain XML → nuclide network |
| `decay_preprocessor/bateman_solver.py` | Time-integrates nuclide populations |
| `decay_preprocessor/decay_fitter.py` | Fits 3-term exponential to the computed decay curve |

---

## 6. Decay Preprocessor Workflow

The preprocessor is a self-contained tool that runs **once** to produce the
`decay_terms` for `config.yaml`.  It is not called by the thermal solver at
run time.

### Inputs

1. **Isotope inventory CSV** — columns `Isotope` and `Atoms`.  Lists the number
   of atoms of each nuclide present in the waste glass at `t = 0` (immediately
   after reactor discharge or reprocessing).

2. **OpenMC decay chain XML** — defines every nuclide's half-life, decay modes,
   branching ratios, and mean decay energy.  The file `chain_endfb71_pwr.xml`
   (ENDF/B-VII.1, PWR) covers ~1600 nuclides and is included in this repository.

### Processing steps

**Step 1 — Parse chain** (`chain_parser.py`)
Reads the XML and builds three arrays:
- `decay_constants[i]` = λᵢ = ln2 / t½  [1/s]
- `q_values[i]` = mean decay energy per disintegration [eV]
- `matrix_A` = sparse Bateman transmutation matrix (N × N)

The matrix encodes the full decay network: diagonal entries are `−λᵢ` (loss from
decay), off-diagonal entries are `+λᵢ · branching_ratio` (production of daughters).

**Step 2 — Load inventory**
Maps each `Isotope` name in the CSV to the chain's index to build the initial
condition vector `N₀`.

**Step 3 — Solve Bateman equations** (`bateman_solver.py`)
Integrates `dN/dt = A · N` forward in time using a stiff BDF solver.  At each
time point the specific power is:

```
Q(t) = (1/mass) · Σᵢ  λᵢ · Nᵢ(t) · qᵢ · eV_to_J
```

Output is a CSV with columns `Time_Years`, `Heat_Watts`, `Specific_Power_W_kg`
on a log-spaced grid spanning the requested duration.

**Step 4 — Fit exponentials** (`decay_fitter.py`)
Fits the 3-term model to the Bateman-computed `Q(t)` curve.  The first month of
data is excluded — very short-lived nuclides dominate the early peak but are
irrelevant to cooling schedules on the scale of years.  The fit uses linear-space
(absolute) residuals and `scipy.optimize.curve_fit`.

Prints R² and RMSE, and outputs the 6 parameters formatted for direct paste into
`config.yaml`.

---

## 7. Configuration Reference

All parameters live in `config.yaml`.  Key decisions to make for a new case:

### Temperature limits and safety margin

```yaml
centerline_limit_C: 400.0    # Glass transition limit [°C]
safety_factor: 1.0           # Effective limit = nominal / safety_factor
surface_limits_C:
  Bentonite: 100.0            # Max canister surface temp in Bentonite buffer [°C]
  Salt: 200.0                 # Max canister surface temp in salt rock [°C]
```

Set `safety_factor ≥ 1.25` to apply a 20% engineering margin to all limits.

### Material properties

```yaml
waste_form:
  rho_base: 2500.0               # Base glass density [kg/m³]
  cp_poly:  [C0, C1, C2]        # Cₚ(T) = C0 + C1·T + C2·T²  [J/(kg·K)]
  k_poly:   [C0, C1, C2]        # k(T)  = C0 + C1·T + C2·T²  [W/(m·K)]
  decay_terms:                   # Output from the decay preprocessor
    - [A1, lambda1]
    - [A2, lambda2]
    - [A3, lambda3]
```

`T` in the polynomials is always in **Kelvin**.

### Sweep and feasibility

```yaml
radii_min: 0.01          # [m]
radii_max: 0.8           # [m]
radii_steps: 100         # Number of log-spaced points
loadings_pct: [5, 10, 15, 20]   # Waste loading fractions to evaluate

max_h_active: 200.0      # Points above this HTC are flagged infeasible
max_cooling_years: 10.0  # Points above this cooling time are flagged infeasible
```

---

## 8. Output Interpretation

### CSV file

`results/Design_Envelope_<name>_<repo>.csv` has one row per (loading, radius)
combination:

| Column | Meaning |
|---|---|
| `Loading_pct` | Waste loading [%] |
| `Radius_m` | Canister outer radius [m] |
| `Min_H_Active_W_m2_K` | Minimum active HTC needed [W/(m²·K)] |
| `Min_Cooling_Years` | Minimum pre-emplacement cooling time [years] |

A value of `0.0` for `Min_H_Active` means passive cooling at the surface is
already sufficient (the peak temperature never exceeds the limit even at h → 0).

### Design envelope plot

Two vertically stacked subplots sharing the x-axis (canister radius):

- **Top:** Required active HTC vs radius, one curve per loading %.  Curves that
  reach `max_h_active` are infeasible with practical equipment.
- **Bottom:** Required cooling years vs radius, one curve per loading %.  Points
  that reach `max_cooling_years` indicate the canister needs longer-than-acceptable
  active storage.

Both curves are monotonically decreasing with radius (larger canister → lower
surface-area-to-volume ratio → harder to cool → stricter requirements).

---

## 9. Key Assumptions and Limitations

- **1D radial symmetry.** The solver treats the canister as an infinite cylinder:
  no axial heat conduction.  This is conservative (underestimates cooling) for
  short canisters.

- **No steel canister wall.** The thermal resistance of the canister wall is
  neglected.  For stainless steel (~15 W/(m·K)) and typical wall thicknesses this
  is a small correction.

- **Constant ambient temperature.** The coolant/repository temperature `T_∞` is
  fixed.  Thermal interaction between adjacent canisters is not modelled.

- **Single waste form.** The solver evaluates one set of material properties and
  one decay curve per run.  Comparing multiple waste forms requires separate runs
  with different `config.yaml` settings.

- **3-term exponential decay.** The fitter uses exactly three terms.  For decay
  curves with very sharp early peaks or very long-lived tails a 3-term fit may
  not capture the full dynamic range perfectly — check the R² and RMSE output by
  the preprocessor.

- **Constant material properties in the passive phase.** The analytical steady-
  state solution uses fixed `k` and `h`.  The values used are those evaluated at
  ambient temperature.  Temperature-dependent variation during the passive phase
  is not captured.

---

## 10. Extending the Solver

### Adding a new repository type

Add an entry to `surface_limits_C` in `config.yaml`:

```yaml
surface_limits_C:
  Bentonite: 100.0
  Salt: 200.0
  Granite: 85.0        # ← new
```

Then run with `--repo Granite`.

### Changing the decay heat model

Run the preprocessor on a new inventory or with a different chain file.  Paste
the printed `decay_terms` into `config.yaml` under `waste_form`.

### Changing material properties

Replace the `cp_poly` and `k_poly` coefficients.  The polynomials are evaluated
in Kelvin at every finite-difference node and time step — no code changes are
needed.

### Increasing solver accuracy

Increase `nodes` (radial resolution) and decrease solver tolerances (edit
`DEFAULT_RTOL` in `src/constants.py`).  This increases run time roughly linearly
with `nodes`.
