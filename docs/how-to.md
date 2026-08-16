# How-to guides

Short recipes for specific tasks. Each is self-contained; read only the one you
need.

Assumes you have run through [tutorial.md](tutorial.md) once. For what every
setting means, see [reference.md](reference.md).

- [Check specific canister designs](#check-specific-canister-designs)
- [Compare only some cooling technologies](#compare-only-some-cooling-technologies)
- [Use my own isotope inventory](#use-my-own-isotope-inventory)
- [Add my own waste form material](#add-my-own-waste-form-material)
- [Compare two matrices](#compare-two-matrices)
- [Add a repository geology](#add-a-repository-geology)
- [Define my own cooling technology](#define-my-own-cooling-technology)
- [Model an LWR operator with years of pool storage](#model-an-lwr-operator-with-years-of-pool-storage)
- [Model rapid encapsulation, weeks after shutdown](#model-rapid-encapsulation-weeks-after-shutdown)
- [Work from a sample rather than the whole campaign](#work-from-a-sample-rather-than-the-whole-campaign)
- [Test sensitivity to the safety factor](#test-sensitivity-to-the-safety-factor)
- [Reproduce a run from its results](#reproduce-a-run-from-its-results)
- [Speed up a slow run](#speed-up-a-slow-run)
- [Use AETHON from Python](#use-aethon-from-python)
- [Diagnose a run that returns nothing](#diagnose-a-run-that-returns-nothing)

---

## Check specific canister designs

The maps are for exploring. Once you have a shortlist, name it in
`solver_config.yaml`:

```yaml
candidates:
  - {name: A, radius_m: 0.080, loading_pct: 15}
  - {name: D, radius_m: 0.215, loading_pct: 25}
```

Every run then prints a table of those designs against each geology and
cooling technology, and writes `candidates_<material>.csv`:

```
Name  Radius_m  Loading_Pct  N_canisters Geology Archetype  t_encap_yr  t_coolers_off_yr  t_active_yr  t_geo_yr  Min_H_Active
   A     0.080       15.000           30    Salt ForcedAir       0.192             0.584        0.392     1.466        25.092
   D     0.215       25.000            1    Salt ForcedAir       1.413            41.110       39.697    79.904        25.152
```

Read it as a timeline. Design A: seal at 2.3 months, run coolers for
5 months, emplace at 1.5 years, and build 30 canisters. Design D is one
canister but waits 80 years.

Radii need not lie on the sweep grid — a real proposal rarely coincides with a
log-spaced point.

`Min_H_Active` is the convective performance the facility would actually have
to deliver, quoted against `T_ambient_active_C`. Use it when none of the named
technologies matches your plant. It is held a degree below the temperature
limit, so expect it slightly above the `h` of a technology that only just
copes — that is the margin, not a contradiction.

---

## Use my own isotope inventory

Write a CSV with two columns:

```csv
Isotope,Atoms
Cs137,3.00e+25
Sr90,8.50e+24
```

Nuclide names must match your chain file's spelling (`Cs137`, not `Cs-137`).
Lines starting `#` are ignored, so keep a provenance header if you have one.

Then:

```bash
decay-preprocessor \
    --inventory   path/to/your_inventory.csv \
    --chain       data/chain_endfb71_pwr.xml \
    --output-dir  results/decay
```

Check two lines in the output. The match count — unmatched names are dropped
**silently**, so `280 of 314 matched` means 34 isotopes disappeared without an
error. And the reported waste mass, computed from your atom counts; if it is
not roughly what you expect, your inventory is probably incomplete.

Point `waste_source` in `solver_config.yaml` at the resulting
`results/decay/waste_source.yaml`.

---

## Add my own waste form material

Add an entry under `materials:` in `solver_config.yaml`:

```yaml
materials:
  MySynroc:
    rho_base: 4300.0                        # matrix density [kg/m3]
    k:  "2.1 - 8.0e-4*T"                    # W/(m.K), T in KELVIN
    cp: "np.interp(T, [300,600,900], [520,690,780])"   # J/(kg.K)
    centerline_limit_C: 1100.0              # your stability limit
```

Then `aethon --material MySynroc`, or set `material: MySynroc` in the config.

`k` and `cp` are Python expressions in `T` **in Kelvin**, with `numpy` as `np`.
Constants, polynomials, Shomate forms, power laws and `np.interp` tables all
work — see [reference.md](reference.md#material).

Two things to check on any new material:

- **Positive across the range.** A polynomial fit can go negative outside the
  data it was fitted to. Verify `k` and `cp` stay positive from 300 K to well
  above your limit.
- **The right limit.** `centerline_limit_C` is a material property, so it lives
  with the material, not globally.

---

## Compare two matrices

The matrix is fixed for a whole run, so run twice and set the results side by
side:

```bash
aethon --material BorosilicateGlass    --output-dir results/glass
aethon --material CA_Recycling_Bg-CaF2 --output-dir results/ceramic
```

Compare the two sets of maps, or the two `explore_full_*.csv` files.

If the matrices belong to genuinely different waste streams — as the two
Copenhagen forms do — change `waste_source` between runs as well. A matrix
comparison holding the wrong decay curve fixed is meaningless.

---

## Add a repository geology

```yaml
repositories:
  Bentonite:
    surface_limit_C: 100.0
  Salt:
    surface_limit_C: 200.0
  Granite:              # new
    surface_limit_C: 85.0
```

Then add it to the `geologies:` list to include it in runs:

```yaml
geologies: [Bentonite, Salt, Granite]
```

Comment that line out to compare everything defined, or narrow a single run
with `--repo Granite`. Selecting is kept separate from defining so a geology
you are not currently comparing keeps its limit in the file.

The limit is set by the buffer material in contact with the canister, not by
the rock itself.

---

## Define my own cooling technology

The shipped figures are literature-typical ranges for orientation, **not
vendor data**. Replace them with your own facility's numbers before relying on
any feasibility conclusion.

They are written out in the `cooling_archetypes:` block of
`solver_config.yaml`, so edit them in place:

```yaml
cooling_archetypes:
  ForcedAir:
    h: 18.0               # was 25.0
    ambient_C: 45.0       # was 40.0
    description: Fan-driven air, our hall's measured performance.
  MyChiller:              # a technology of your own
    h: 400.0
    ambient_C: 18.0
    description: Circulating chilled water jacket.
```

What is in that block is the whole library — delete an entry and that
technology is gone. To compare a subset without losing definitions, use
[`archetypes:`](#compare-only-some-cooling-technologies) instead.

Both `h` and `ambient_C` are required. They travel as a pair because an HTC
means nothing without the temperature it works against — a cooler hall
substitutes directly for a better coefficient.

Give worst-case values: the hottest ambient and the weakest performance you
would design to. Real operation can then only be better than predicted.

Check what is active with `aethon --list-archetypes`.

---

## Compare only some cooling technologies

`cooling_archetypes` defines the technologies; `archetypes` chooses which of
them a run compares. Use the second one to narrow a run, so the definitions
stay in the file for next time:

```yaml
archetypes: [ForcedAir]     # omit to compare all of them
```

Or for a single run:

```bash
aethon --archetype ForcedAir
```

The flag overrides the config. Both accept several names:
`archetypes: [ForcedAir, WaterPool]`, `--archetype ForcedAir WaterPool`.

This is also the fastest way to cut run time — the transient gate is solved
once per grid point *per technology*, so dropping from three to one is a
three-fold saving.

---

## Model an LWR operator with years of pool storage

```bash
aethon --t-pre-min 5 --t-pre-max 10 --material BorosilicateGlass
```

Expect `t_active = 0` on most or all designs: after five years the waste is
already passively safe on centreline, so the coolers never run. `NaturalAir`
will be feasible.

That is the answer — **no active cooling plant is needed** — and the remaining
question becomes purely how many canisters to trade against time to repository.

---

## Model rapid encapsulation, weeks after shutdown

```bash
aethon --t-pre-min 0.02 --t-pre-max 0.25 --repo Salt
```

Expect weaker technologies to drop out: designs where `Feasible` is false could
not be held below the centreline limit within that window.

**Check your decay curve covers the window.** The fit is only valid from the
preprocessor's `--cutoff-years` (default one month, 0.0833 yr) onward. A
`--t-pre-min` of 0.02 is below that, so re-fit first:

```bash
decay-preprocessor --inventory ... --chain ... \
    --cutoff-years 0.01 --output-dir results/decay
```

Below the cutoff the fitted curve substantially overstates heat, which would
make your designs look worse than they are.

---

## Work from a sample rather than the whole campaign

Nothing special is needed. Specific power is intensive — a kilogram of this
waste runs just as hot whether you have 10 kg or 10 tonnes of it — so the
preprocessor does not care how representative your inventory is:

```bash
decay-preprocessor \
    --inventory   sample.csv \
    --chain       data/chain_endfb71_pwr.xml \
    --output-dir  results/decay
```

Set the campaign size separately, in the config:

```yaml
campaign:
  total_waste_mass_kg: 4000.0
```

or at the command line, without re-fitting anything:

```bash
aethon --total-mass 4000
```

**The one case that needs `--sample-mass`:** an inventory that lists only
*part* of the waste it belongs to — radioactive species with the stable carrier
omitted, say. The computed mass would then be too low and the specific power
too high. Supply the true mass of what the CSV describes:

```bash
decay-preprocessor --inventory radionuclides_only.csv \
    --chain data/chain_endfb71_pwr.xml --sample-mass 12.5
```

The preprocessor prints the inventory-implied mass alongside yours and warns if
they differ by more than a factor of two.

---

## Test sensitivity to the safety factor

The safety factor divides every temperature limit, so it is the natural knob
for "what does extra margin cost me?". Run the same search at several values
into separate directories:

```bash
for sf in 1.0 1.1 1.25; do
    aethon --safety-factor $sf --output-dir results/sf_$sf
done
```

A factor of 1.25 turns a 500 degC limit into 400 degC, which shows up as later
milestones, more canisters, or designs dropping out as infeasible.

Each directory carries a `run_config.yaml` recording which value produced it,
so the comparison stays identifiable afterwards.

---

## Reproduce a run from its results

Every output directory contains `run_config.yaml`, holding the settings that
produced it. Its header gives the command:

```bash
head -6 results/my_run/run_config.yaml
```

```
#   Reproduce with:
#     aethon --config run_config.yaml
```

Run it from that directory, or pass the path:

```bash
aethon --config results/my_run/run_config.yaml \
    --output-dir results/my_run_again
```

No flags are needed. The file is self-contained — decay terms and material
coefficients are written out in full rather than referenced, and the geologies
and cooling technologies that were compared are recorded as selections — so it
reproduces correctly even if the inventory or decay curve it came from has
since been regenerated.

---

## Speed up a slow run

In rough order of effect:

Cost is `radii.steps x loadings.steps x archetypes` transient solves. Geology
is free — the encapsulation gate does not depend on it — so narrowing
`geologies` will not speed anything up.

```bash
# Fewer grid points
aethon --radii-steps 10 --loadings-steps 5

# One technology instead of every one
aethon --archetype ForcedAir

# Skip figures (the sweep still runs; only plotting is skipped)
aethon --no-plot
```

The run prints the count up front:

```
Grid: 20 radii x 11 loadings = 220 designs
Transient gate solves: 660
```

A coarser grid makes the maps look stepped. If you only care about the passive
milestones — `t_coolers_off` and `t_geo` — those are analytic and cost almost
nothing, so you can raise the grid freely and cut the technologies instead.

Lowering `nodes` in the config (default 50) also helps, but it coarsens the
radial discretisation and changes results. Reduce grid density first.

---

## Use AETHON from Python

Nothing in the package prompts or prints unless you ask it to — `verbose=False`
suppresses the progress bar, and the console layer is only reached through the
CLI.

```python
from aethon import load_config, run_exploration

cfg = load_config("solver_config.yaml", material="CA_Recycling_Bg-CaF2")
full_df = run_exploration(
    cfg=cfg,
    repositories=["Salt"],
    archetype_names=["ForcedAir"],
    verbose=False,
)

print(full_df[["Radius_m", "Loading_Pct", "N_canisters",
               "t_encap_yr", "t_geo_yr"]])
```

`run_exploration` returns a single long-format frame with every point in it.
To get a 2D field for your own plotting:

```python
from aethon.design.report import pivot_field

radii, loadings, t_geo = pivot_field(full_df, "t_geo_yr")
```

To evaluate designs of your own without sweeping:

```python
from aethon.design.candidates import evaluate_candidates

table = evaluate_candidates(
    cfg,
    repositories={"Salt": 200.0},
    archetypes={"ForcedAir": {"h": 25.0, "ambient_C": 40.0}},
    candidates=[{"name": "A", "radius_m": 0.08, "loading_pct": 15}],
)
```

Override config values by assigning to `cfg` before the call:

```python
cfg["pre_encap_max_years"] = 10.0
cfg["total_waste_mass_kg"] = 4000.0
```

Lower-level entry points, if you want one milestone rather than a search:

```python
from aethon.analysis.pipeline import (
    find_total_decay_years,   # t_coolers_off / t_geo
    find_min_encap_years,     # t_encap for a given technology
)
```

---

## Diagnose a run that returns nothing

`No feasible design found` means every candidate failed. Work through these:

**1. Check the header block.** Every run prints what it is modelling:

```
Material:     CA_Recycling_Bg-CaF2
Waste stream: results/decay/waste_source.yaml
              Q(0) = 1683.11 W/kg
```

If it says `inline decay_terms` you are running on placeholder data.

**2. Is the encapsulation window too narrow?** If no technology can cool the
waste within `pre_encapsulation_years.max`, everything is infeasible. Widen it
and see what appears.

**3. Are the radii too large?** The `Q·R²/(4k)` term means big canisters cannot
be cooled by any surface treatment. Try `--radii-max 0.2`.

**4. Is the loading too high?** Try `--loadings 5`.

**5. Is the safety factor doing it?** `safety_factor: 1.25` cuts a 500 °C limit
to 400 °C. Set it to 1.0 temporarily to see whether that alone is the cause.

To find where feasibility breaks, look at `explore_full_*.csv` — the sweep is
exhaustive, so infeasible designs are still in there with `Feasible` false and
`t_encap_yr` infinite. The shaded region of the encapsulation map is exactly
those rows.
