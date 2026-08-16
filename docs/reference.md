# Reference

Complete listing of commands, configuration keys, file formats, and outputs.
For a guided first run see [tutorial.md](tutorial.md); for the reasoning behind
the model see [physics.md](physics.md).

**Contents**

- [Commands](#commands) — [`aethon`](#aethon), [`decay-preprocessor`](#decay-preprocessor)
- [Configuration file](#configuration-file)
- [Waste mass](#waste-mass)
- [Cooling technologies](#cooling-technologies)
- [Bundled materials](#bundled-materials)
- [File formats](#file-formats)
- [Output columns](#output-columns)
- [Conventions and limits](#conventions-and-limits)

---

## Commands

### `aethon`

Sweeps canister designs and writes the design-space maps. Also runnable as
`python -m aethon`.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--config` | path | `./solver_config.yaml` | Configuration file to read |
| `--material` | name | from config | Matrix to use, from the `materials` library |
| `--repo` | name(s) | from config, else all | Repository geologies to evaluate |
| `--archetype` | name(s) | from config, else all | Cooling technologies to evaluate |
| `--t-pre-min` | float | from config | Earliest encapsulation [yr from shutdown] |
| `--t-pre-max` | float | from config | Latest acceptable encapsulation [yr from shutdown] |
| `--total-mass` | float | from config | Campaign waste mass [kg] |
| `--safety-factor` | float | from config | Divisor on all temperature limits |
| `--aspect-ratio` | float | from config | Canister height / radius |
| `--radii-min` | float | from config | Smallest canister radius [m] |
| `--radii-max` | float | from config | Largest canister radius [m] |
| `--radii-steps` | int | from config | Number of radii (log-spaced) |
| `--loadings-min` | float | from config | Smallest waste loading [wt%] |
| `--loadings-max` | float | from config | Largest waste loading [wt%] |
| `--loadings-steps` | int | from config | Number of loadings (linear) |
| `--loadings` | float(s) | from config | Explicit loadings, overriding the range |
| `--output-dir` | path | `results` | Where CSVs and figures are written |
| `--no-plot` | flag | off | Skip figure generation |
| `--list-archetypes` | flag | off | Print the cooling technology library and exit |

The loading grid is resolved in this order, first match winning:

1. `--loadings` on the command line
2. any `--loadings-min/max/steps` on the command line
3. the `loadings:` block in the config

`--loadings` is for quick runs only. Three or four values is too coarse to
draw contours through, so the maps will look stepped.

`--repo` and `--archetype` take several names: `--repo Bentonite Salt`.

Every run begins by printing what it is modelling — material, waste stream and
its `Q(0)`, campaign mass, encapsulation window. Check that block before
trusting results; it is what reveals a config still pointing at placeholder
decay data.

**Exit codes:** `0` success, `1` a bad argument or unreadable configuration.

### `decay-preprocessor`

Converts an isotope inventory into a fitted decay curve. Also runnable as
`python -m decay_preprocessor.run_preprocessor`.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--inventory` | path | **required** | Isotope inventory CSV |
| `--chain` | path | **required** | OpenMC-format decay chain XML |
| `--sample-mass` | float | computed | Override the waste mass [kg] — see [Waste mass](#waste-mass) |
| `--duration` | float | `50.0` | Years to simulate; bounds where the fit is valid |
| `--cutoff-years` | float | `1/12` | Ignore data before this time when fitting |
| `--n-points` | int | `2000` | Log-spaced time points in the solve |
| `--output-dir` | path | `.` | Where outputs are written |
| `--no-waste-source` | flag | off | Print terms only; skip `waste_source.yaml` |

**Exit codes:** `0` success, `1` no isotopes matched the chain.

---

## Configuration file

Default location `./solver_config.yaml`; override with `--config`.

### What belongs in the config, and what belongs on the command line

The config describes **your situation** — things that change rarely and that
you would commit to version control. Flags ask **a question about it** — things
that change from run to run. Where both exist, the flag overrides the config
for that run only.

| Setting | Config | Flag | Why |
|---|---|---|---|
| `materials`, `repositories`, `cooling_archetypes` | yes | — | The three libraries. Structured data, unusable as flags |
| `material` | yes | `--material` | Which matrix |
| `geologies` | yes | `--repo` | Which geologies to compare |
| `archetypes` | yes | `--archetype` | Which technologies to compare |
| `waste_source` / `decay_terms` | yes | — | Six numbers plus provenance |
| `campaign.total_waste_mass_kg` | yes | `--total-mass` | |
| `campaign.canister_aspect_ratio` | yes | `--aspect-ratio` | |
| `pre_encapsulation_years` | yes | `--t-pre-min/max` | The main "what if" dial |
| `safety_factor` | yes | `--safety-factor` | The main sensitivity knob |
| `radii`, `loadings` | yes | `--radii-*`, `--loadings-*` | Sweep resolution |
| `candidates` | yes | — | A shortlist you are checking; structured data |
| `passive` | yes | — | Site conditions; do not vary run to run |
| `nodes`, `max_years` | yes | — | Solver internals, not physics questions |

Every run writes the resolved settings — config plus flag overrides — to
[`run_config.yaml`](#run_configyaml) beside the results.

### Material

```yaml
material: BorosilicateGlass        # which entry below to use

materials:
  BorosilicateGlass:
    rho_base: 2230.0               # matrix density [kg/m3]
    k:  "1.2"                      # thermal conductivity [W/(m.K)]
    cp: "935.56 + 0.38953*T - 24617000/T**2"   # specific heat [J/(kg.K)]
    centerline_limit_C: 500.0      # devitrification onset [degC]
```

`k` and `cp` are Python expressions in `T`, **temperature in Kelvin**, with
`numpy` available as `np`. The result is broadcast to `T`'s shape, so a
constant such as `"1.2"` is valid. Examples:

| Form | Example |
|---|---|
| Constant | `"1.2"` |
| Polynomial | `"500.0 + 0.5*T - 1e-4*T**2"` |
| Shomate | `"935.56 + 0.38953*T - 24617000/T**2"` |
| Power law | `"200.0 * T**0.35"` |
| Tabulated | `"np.interp(T, [300,400,500], [450,500,540])"` |

`centerline_limit_C` is a property of the matrix, so it lives with the matrix.

### Waste stream

```yaml
waste_source: results/decay/waste_source.yaml   # preferred

decay_terms:                                    # fallback, used only if
  - [100.0, 5.0]                                # waste_source is absent
  - [ 20.0, 0.5]
  - [  2.0, 0.05]
```

`waste_source` points at preprocessor output and takes precedence over inline
`decay_terms`. Relative paths resolve against the **config file's** directory,
not the working directory.

`decay_terms` is a list of `[amplitude_W_per_kg, decay_constant_per_yr]` pairs
giving `Q(t) = sum(A_i * exp(-lambda_i * t))` with `t` in years.

### Campaign

```yaml
campaign:
  total_waste_mass_kg: 116.0     # total waste to encapsulate [kg]
  canister_aspect_ratio: 6.0     # height / radius
```

`total_waste_mass_kg` is overridden by `waste_source` when that file records
it. Affects fleet size only, never temperature.

### Pre-encapsulation window

```yaml
pre_encapsulation_years:
  min: 0.0833      # earliest you can deliver waste [yr from shutdown]
  max: 5.0         # latest you are willing to wait
```

The solver finds the earliest feasible encapsulation inside this window for
each technology. A technology needing longer than `max` is infeasible.

`min` must not be earlier than the preprocessor's `--cutoff-years`, or the
decay curve is being read outside its fitted range.

### Safety, site, and geologies

```yaml
safety_factor: 1.0     # divisor on ALL temperature limits

passive:               # worst-case design basis for both passive phases
  ambient_C: 50.0      # interim store / repository temperature [degC]
  h: 5.0               # natural convection HTC [W/(m2.K)]

repositories:
  Bentonite:
    surface_limit_C: 100.0
  Salt:
    surface_limit_C: 200.0
```

Effective limit is `nominal / safety_factor`, so `1.25` gives 20% margin.

There is deliberately **no global ambient temperature**. The active facility's
ambient belongs to the cooling technology (see below); only the passive phases
have a site-wide value.

### Cooling technologies (optional)

Two keys, doing two different jobs. Mixing them up is the easiest mistake to
make here.

**`archetypes` selects** which technologies a run compares:

```yaml
archetypes: [ForcedAir, WaterPool]
```

Omit it to compare every technology defined. `--archetype` overrides it, like
every other flag.

**`cooling_archetypes` defines** them. The shipped config writes the library
out in full, so editing a technology means changing the number in place:

```yaml
cooling_archetypes:
  NaturalAir:
    h: 5.0                       # [W/(m2.K)]
    ambient_C: 40.0              # [degC]
    description: Unforced air in a passive vault; no cooling plant.
  ForcedAir:
    h: 25.0
    ambient_C: 40.0
    description: Fan-driven air over the canister surface.
  WaterPool:
    h: 750.0
    ambient_C: 40.0
    description: Full immersion in a cooled water pool.
```

What you write here is what exists — the same as `materials` and
`repositories`. Deleting an entry removes that technology; adding one adds it.
To compare only some of them without losing definitions, use `archetypes`
above rather than deleting.

Omitting the block entirely falls back to a copy of this library held in
`aethon/design/archetypes.py`, so the package still works from Python without
a config file.

`h` and `ambient_C` are required on every entry; `description` is optional and
only shown by `--list-archetypes`. The two figures travel as a pair because
convective flux is `h*(T_surface - T_ambient)` — an HTC means nothing without
the temperature it works against, and a cooler hall substitutes directly for a
better coefficient.

### Sweep and solver

```yaml
radii:                          # log-spaced
  min: 0.05                     # [m]
  max: 0.5                      # [m]
  steps: 20

loadings:                       # linear
  min: 5.0                      # [wt%]
  max: 25.0                     # [wt%]
  steps: 11

nodes: 50                       # radial finite-difference nodes
max_years: 50.0                 # transient simulation horizon [yr]
```

Every range in the config has this shape — a block with `min`, `max` and,
where it is a grid, `steps`. `pre_encapsulation_years` is the same, without
`steps`.

These are the axes of both output maps, so they need enough points to draw
smooth contours through. Run time scales as
`radii.steps * loadings.steps * archetypes`; the passive map is analytic and
effectively free at any resolution, and only the encapsulation gate is
expensive.

### Candidates (optional)

```yaml
candidates:
  - {name: A, radius_m: 0.080, loading_pct: 15}
  - {name: D, radius_m: 0.215, loading_pct: 25}
```

Designs you have shortlisted. Each is evaluated exactly against every selected
geology and cooling technology and reported in a table and
`candidates_<material>.csv`. They need **not** lie on the sweep grid — the
point is to check a design somebody has proposed, which rarely coincides with
a log-spaced radius.

`name` is optional and defaults to `C1`, `C2`, ... A malformed entry stops the
run rather than being skipped: a silently dropped candidate looks exactly like
one that was evaluated and found unremarkable.

---

## Waste mass

Heat output is normalised to W/kg using a mass derived from the inventory
itself:

```
m = sum_i (N_i / N_A) * A_i
```

Since that mass comes from the same atom counts that produce the heat, the two
cannot disagree. No mass need be supplied.

Mass number `A` stands in for true atomic mass. The difference is the nuclear
binding energy, about 0.1 %, which is negligible here.

**When the computed value is wrong.** It is the whole waste mass only if the
inventory lists *every* nuclide present, stable ones included. An inventory of
radioactive species alone gives too small a mass and therefore too high a
specific power. The computed value is always printed, so an unexpected figure
is visible immediately:

```
      314 isotopes in inventory, 314 matched to chain.
      Waste mass: 115.984 kg
```

**`--sample-mass`** overrides it, for exactly that case. The preprocessor
reports the inventory-implied value alongside, and warns when the two differ by
more than a factor of two — the signature of supplying a canister or
glass-composite mass rather than a waste mass.

Every row contributes mass, whether or not it matched the decay chain; an
unmatched nuclide is still mass in the canister. The match count line is what
polices naming errors.

### Campaign mass is a separate thing

How much waste exists in total does not affect specific power, which is
intensive. It is set only in the configuration:

```yaml
campaign:
  total_waste_mass_kg: 116.0
```

or with `aethon --total-mass`. The preprocessor has nothing to do with it.

### Why it must be waste mass, not composite mass

The solver computes volumetric heat as

```
Q_vol = decay(t) [W/kg] * rho_eff [kg/m3] * loading_fraction
```

For that to yield W/m3, `decay(t)` must be per kilogram of **waste**. The
matrix dilution is already accounted for by `rho_eff` and `loading_fraction`,
so a composite mass would double-count it. Deriving the mass from the
inventory's nuclides gives the waste mass by construction.

---

## Cooling technologies

You choose a technology, not a heat transfer coefficient. Each carries a
worst-case design-basis `(h, ambient_C)` pair, because an HTC is meaningless
without the temperature it works against, and nobody can state the ambient of
a facility that does not exist yet.

| Name | `h` [W/(m2.K)] | `ambient_C` [degC] | Description |
|---|---|---|---|
| `NaturalAir` | 5 | 40 | Unforced air in a passive vault; no cooling plant |
| `ForcedAir` | 25 | 40 | Fan-driven air over the canister surface |
| `WaterPool` | 750 | 40 | Full immersion in a cooled water pool |

> **These are literature-typical convective ranges for orientation, not vendor
> performance data.** Replace them with figures from your own facility design
> before relying on any feasibility conclusion.

The three built-ins happen to share an ambient of 40 degC, but the pairing is
not decorative: convective flux is `h*(T_surface - T_ambient)`, so a cooler
hall substitutes directly for a better coefficient. A technology defined at
250 W/(m2.K) and 25 degC can outperform one at 750 W/(m2.K) and 40 degC. Always
state both when you define your own.

Print the active library with `aethon --list-archetypes`.

---

## Bundled materials

| Name | rho_base [kg/m3] | Notes |
|---|---|---|
| `BorosilicateGlass` | 2230 | Reference matrix for LWR and conventional reactor waste |
| `CA_Recycling_Bg-CaF2` | 2321.5 | HIPed glass-ceramic, borosilicate + CaF2. Thorium MSR, recycling scenario |
| `CA_Emergency_Bg-CaF2-Zr` | 3052.8 | As above with zirconia; denser and more conductive |

All three carry `centerline_limit_C: 500.0`.

The two Copenhagen forms pair with specific *waste streams* as well as being
specific matrices — matching decay curves are in `examples/`. Their `cp` fits
turn over above roughly 1500 K and must not be extrapolated; this is far
outside the 500 degC operating limit.

---

## File formats

### Isotope inventory CSV

Two columns. Lines beginning `#` are ignored, so a header block is fine.
Additional columns are ignored.

```csv
# Any notes you like
Isotope,Atoms
Cs137,3.00e+25
Sr90,8.50e+24
```

| Column | Meaning |
|---|---|
| `Isotope` | Nuclide name, spelled as in the chain file (`Cs137`, not `Cs-137`) |
| `Atoms` | Number of atoms at t = 0 |

Names that do not match the chain are **dropped silently**. The preprocessor
prints how many matched; check that line.

### Decay chain XML

OpenMC format, either a `<depletion_chain>` root with `decay_energy` on each
`<nuclide>`, or the older `<chain>` root with `energy` on each `<decay>`.
Download from <https://openmc.org/nuclear-data/>. Roughly 27 MB; not bundled.

### `waste_source.yaml`

Generated by the preprocessor. Regenerate rather than editing.

```yaml
decay_terms:
- [913.012, 22.9854]
- [725.76, 3.84445]
- [44.3378, 0.0231816]
sample_mass_kg: 115.984
fit_r_squared: 0.9999966548937562
fit_rmse_W_per_kg: 0.3301081814006397
total_waste_mass_kg: 115.984
source_inventory: examples/msr_inventory_5y.csv
source_chain: data/chain_endfb71_pwr.xml
```

`decay_terms` and `total_waste_mass_kg` are consumed by the solver; the rest is
provenance.

---

## Output columns

`explore_full_<material>.csv` holds every grid point — one row per (radius,
loading, geology, technology). The sweep is exhaustive: nothing is filtered,
ranked or thinned, so a design that no technology can cool is still present
with `Feasible` false.

**All times are years from reactor shutdown.**

| Column | Meaning |
|---|---|
| `Geology` | Repository geology evaluated |
| `Archetype` | Cooling technology evaluated |
| `Material` | Matrix used |
| `Radius_m` | Canister outer radius [m] |
| `Loading_Pct` | Waste loading [wt%] |
| `N_canisters` | Fleet size |
| `t_encap_yr` | Earliest encapsulation; active cooling begins. `inf` where infeasible |
| `t_coolers_off_yr` | Centreline passively safe; coolers off, move to interim store |
| `t_active_yr` | `t_coolers_off - t_encap`; how long the plant actually runs |
| `t_geo_yr` | Repository emplacement allowed |
| `Binding_At_Geo` | `surface` or `centre`; which limit governed at emplacement |
| `Feasible` | Whether the technology can cool this design within the window |
| `h_active` | Chosen technology's HTC [W/(m2.K)] |
| `T_ambient_active_C` | Chosen technology's facility ambient [degC] |
| `h_passive` | Passive HTC [W/(m2.K)] |
| `T_ambient_passive_C` | Passive ambient [degC] |
| `T_peak_centreline_C` | Peak centreline temperature during active cooling [degC] |
| `T_peak_surface_C` | Surface temperature at that moment [degC] |
| `Q_per_canister_W` | Heat output of one canister at `t_encap` [W] |
| `Facility_Duty_W` | `Q_per_canister_W * N_canisters` [W] |

`t_encap` does not vary with geology, and `N_canisters`, `t_coolers_off` and
`t_geo` do not vary with cooling technology. Those repetitions in the CSV are
the join, not duplicated work — each is solved once.

### `candidates_<material>.csv`

Written when the config defines a `candidates:` block. Same columns as above
for each (candidate, geology, technology), plus:

| Column | Meaning |
|---|---|
| `Name` | The name you gave the design |
| `Min_H_Active` | Least HTC that would suffice, at `T_ambient_active_C`. `NaN` where passive convection alone is enough |

`Min_H_Active` is what to specify against if none of the named technologies
matches your facility. It targets one degree **below** the centreline limit
before the safety factor applies, so it is a coefficient you could build to
rather than the critical value at which the design sits exactly on the limit.
Expect it to come out slightly *above* the `h` of a technology that only just
manages the design. It is meaningless without `T_ambient_active_C` beside it.

### `run_config.yaml`

Written into every output directory, recording the settings that produced the
results — config values after any flag overrides. Without it a results
directory cannot be identified from its own contents: the CSVs carry the design
and its operating conditions, but not the pre-encapsulation window, safety
factor, campaign mass, or which decay curve was used.

It is **self-contained** — decay terms and material coefficients are written out
in full rather than referenced by path, so the record stays truthful even if
`waste_source.yaml` is later regenerated from a different inventory.

It is **re-runnable**. The header carries the exact command:

```yaml
# Settings that produced the results in this directory.
# Written by AETHON on 2026-08-12 16:00 UTC.
#
#   Reproduce with:
#     aethon --config run_config.yaml
#
# Decay curve originally from: examples/ca_recycling_waste_source.yaml
```

Both enumerated dimensions round-trip, by different routes. `repositories`
defines the geologies, so writing it is enough. `cooling_archetypes` only
overrides entries in the built-in library, so the selection is written
separately as `archetypes`. Without that a rerun would widen to every
technology and produce more rows than the results it claims to describe.

The reproduce command therefore needs no flags.

### Figures

Both are contour maps over the same axes — canister radius (log) against waste
loading — so they can be read against one another.

| File | Content |
|---|---|
| `design_map_passive_<material>.png` | `t_coolers_off`, then `t_geo` for each geology. One panel each |
| `design_map_encapsulation_<material>.png` | `t_encap`, one panel per cooling technology. Shaded where the technology cannot cool the design within the window |

Contour levels are round numbers (0.25 yr, 1 yr, 5 yr, ...) chosen by log
spacing across the data, and are **shared across the panels of a figure** so
the same isoline means the same thing in each. A level outside a given panel's
range simply does not appear there.

Neither figure has an archetype dimension in the first case or a geology
dimension in the second, because neither quantity depends on it. That is what
keeps this to two figures rather than one per combination.

`N_canisters` is not plotted. It is a closed-form function of radius and
loading with no physics in it, and it is reported per design in the CSVs and
the candidates table.

If the contours look stepped, the grid is too coarse — raise `radii.steps` and
`loadings.steps`. The passive map costs almost nothing to refine; the
encapsulation map costs one transient solve per point per technology.

---

## Conventions and limits

| Convention | Value |
|---|---|
| Internal temperature unit | Kelvin (Celsius in config and output) |
| Time unit | Years |
| Effective density | `rho_base / (1 - loading_fraction)` |
| Canister volume | `pi * R^2 * (aspect_ratio * R)` |
| Radius sampling | Logarithmic |
| HTC search bounds | `[h_passive, 2000]` W/(m2.K) |
| Decay-time search bound | `[0, 1000]` years |
| Fleet size rounding | Up; a partly filled canister is still a canister |

**Dependencies:** numpy, scipy, pandas, matplotlib, pyyaml.

**Validity of the decay curve.** The fit is trustworthy between
`--cutoff-years` and `--duration`. Below the cutoff it overstates heat
substantially; beyond the duration it extrapolates. If a result reports
`t_geo` beyond the fitted duration, re-run the preprocessor with a longer one.
