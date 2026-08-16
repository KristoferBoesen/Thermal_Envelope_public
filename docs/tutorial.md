# Tutorial: from an isotope inventory to a cooling schedule

A single worked run, start to finish, using data shipped with the repository.
Follow it exactly and your numbers will match the ones printed here.

Roughly 20 minutes, most of it a one-off download.

By the end you will have two maps of the canister design space, and exact
numbers for three specific designs — how long the cooling plant must run for
each, and when the waste can go in the ground.

> This tutorial has one path and no choices. For alternatives see
> [how-to.md](how-to.md); for what every setting means see
> [reference.md](reference.md); for why the model works this way see
> [physics.md](physics.md).

---

## Step 1 — Install

```bash
git clone <repository-url>
cd aethon
pip install -e .
```

This gives you two commands: `aethon` and `decay-preprocessor`. Check:

```bash
aethon --list-archetypes
```

You should see four cooling technologies listed.

Run `aethon` from the repository directory — it reads `solver_config.yaml` from
the current directory, or from `--config` if you give one.

---

## Step 2 — Download a decay chain

The preprocessor needs nuclear data: half-lives, decay energies, and which
nuclide decays into which. These files are about 27 MB, so they are **not
shipped with the repository** — you download one once and keep it.

Get an OpenMC-format chain from <https://openmc.org/nuclear-data/>. This
tutorial uses `chain_endfb71_pwr.xml` (ENDF/B-VII.1, PWR).

Put it in `data/`, which exists for this purpose and is git-ignored:

```bash
mkdir -p data
# move your downloaded chain_endfb71_pwr.xml into data/
```

Any other chain file works; your fitted numbers will differ slightly from the
ones below.

---

## Step 3 — Look at the inventory

We will use the inventory shipped in `examples/`:

```bash
head -15 examples/msr_inventory_5y.csv
```

```
# ==========================================
#        COPENHAGEN ATOMICS INVENTORY
# ==========================================
# SYSTEM TOTALS:
# Total Mass:  115.984087 kg
# Total Atoms: 1.936783e+27
# ==========================================
Isotope,Atoms
F19,1.5600000000000002e+27
Cs135,3.23e+25
...
```

Two things matter:

- The format is two columns, `Isotope` and `Atoms` (number of atoms at t = 0).
  Lines starting `#` are ignored.
- The header records **115.984087 kg**. You will not have to type that
  anywhere — the preprocessor works it out from the atom counts — but it is
  worth noting so you can check the two agree.

---

## Step 4 — Fit the decay curve

```bash
decay-preprocessor \
    --inventory   examples/msr_inventory_5y.csv \
    --chain       data/chain_endfb71_pwr.xml \
    --output-dir  results/decay
```

No mass is needed. Heat is normalised to watts per kilogram using a mass worked
out from the inventory's own atom counts, which is why the figure it reports
should match the header you just read. If an inventory deliberately lists only
part of the waste, `--sample-mass` overrides it — see
[waste mass](reference.md#waste-mass).

This takes a minute or so. You should see:

```
[1/4] Parsing chain file:  data/chain_endfb71_pwr.xml
      3,819 nuclides loaded.
[2/4] Loading inventory:   examples/msr_inventory_5y.csv
      314 isotopes in inventory, 314 matched to chain.
      Waste mass: 115.984 kg
[3/4] Solving decay chain for 50.0 years ...
[4/4] Fitting sum-of-exponentials ...

============================================================
Fit quality:  R^2 = 0.99999665  |  RMSE = 0.33 W/kg
============================================================

decay_terms (W/kg, 1/yr):

    - [913.012, 22.9854]
    - [725.76, 3.84445]
    - [44.3378, 0.0231816]
```

### Check it worked

Four checks, in order of importance:

**1. Did everything match, and is the mass right?** `314 of 314 matched`, and
`Waste mass: 115.984 kg` against the header's `115.984087 kg`.
Names that do not match the chain are dropped **silently**, so `280 of 314`
would mean 34 isotopes vanished from your problem without an error.

**2. Is the fit good?** R² above 0.999. Here it is 0.99999665.

**3. Does the plot look right?** Open `results/decay/decay_fit.png`. The fitted
line should lie on the computed curve from one month onward. It will diverge
below that — the fitter deliberately ignores the first month so that very
short-lived nuclides, long gone before anything is encapsulated, do not distort
the fit where it matters.

**4. Does the physics look right?** Take the smallest decay constant, 0.0232/yr,
and convert to a half-life: `ln(2)/0.0232 = 29.9 years`. Caesium-137 is 30.1
years and strontium-90 is 28.8 — the two nuclides known to dominate waste heat
decades out. Nothing in the fitter knows that should happen, so when it does,
your chain parsing and decay solve are almost certainly right.

You now have `results/decay/waste_source.yaml`. That one file describes your
waste stream.

---

## Step 5 — Point the solver at it

`solver_config.yaml` ships pointing at a worked example so that a fresh clone
runs immediately. Change that line to your own output:

```yaml
waste_source: results/decay/waste_source.yaml
```

Miss this and you will get perfectly plausible results — for somebody else's
waste. The next step prints which decay curve it used and how hot it is, so
you will see immediately whether the change took effect.

---

## Step 6 — Run the search

```bash
aethon \
    --material CA_Recycling_Bg-CaF2 \
    --radii-min 0.05 --radii-max 0.4 --radii-steps 14 \
    --loadings-min 5 --loadings-max 15 --loadings-steps 6 \
    --repo Salt \
    --archetype ForcedAir \
    --output-dir results/tutorial
```

Two selections are being made, and they are independent:

- **`--material`** picks the *matrix* — the glass or ceramic block, supplying
  conductivity, heat capacity, density and the devitrification limit.
- **`waste_source`** (Step 5) picks the *waste stream* — what is inside it.

The matrix is the container; the waste stream is the contents. We are using the
glass-ceramic that matches this MSR waste. We also restrict to one geology and
one cooling technology to keep the output short; leaving those flags off
compares all of them.

The run opens by stating what it is modelling:

```
Material:     CA_Recycling_Bg-CaF2
Waste stream: results\decay\waste_source.yaml
              Q(0) = 1760.93 W/kg
Campaign:     116 kg
Encapsulation window: 0.0833 to 5 yr from shutdown
```

`Campaign` comes from `campaign.total_waste_mass_kg` in the config, not from
the decay curve — how much waste you have is unrelated to how hot a kilogram of
it runs.

**Check this block.** `Waste stream` should be the file you generated in Step 4.
If it still names `examples/ca_recycling_waste_source.yaml`, Step 5 did not take
effect and you are modelling the shipped example rather than your own waste.

---

## Step 7 — Read the results

The console summarises what was swept:

```
Swept 84 designs
  radius   0.050 to 0.400 m
  loading  5 to 15 wt%

Designs each cooling technology can encapsulate in the window:
  ForcedAir        81 of 84   (96%), earliest t_encap 0.083 yr

Time to repository emplacement:
  Salt           0.33 to 79.15 yr
```

The real output, though, is two figures.

### `design_map_passive_*.png`

Contours over canister radius and waste loading, one panel per milestone:
when the coolers can stop, then when the repository will accept the canister.
Read left to right it is the storage sequence.

Both are properties of the canister and the waste alone — no cooling
technology appears, because neither depends on one.

### `design_map_encapsulation_*.png`

The same axes, contoured with `t_encap` — the earliest the waste can be sealed
into that canister — with one panel per cooling technology. The shaded corner
is where the technology cannot hold the design below the centreline limit
within your encapsulation window at all.

Levels are shared across panels, so the same isoline means the same thing in
each and you can see a stronger technology push its contours to the right.

### What the maps tell you

**Everything gets worse toward the top right.** Bigger radius and higher
loading both mean more heat in a shape that sheds it less easily. Contours run
diagonally because the two trade off against each other.

**The gradient is steep.** On the Salt panel, `t_geo` runs from under a year in
the bottom-left corner to nearly eighty years in the top-right. Canister
geometry is not a detail here; it is the dominant decision.

**The coolers-off contours sit far left of the repository ones.** That gap is
the interim store doing its job — the canister is centreline-safe long before
its surface is cool enough for the buffer, and there is no reason to run
coolers through the wait.

**Small canisters hit a floor.** In the bottom-left of the encapsulation map
`t_encap` stops falling: it has hit 0.083 years, one month, which is the
earliest your configuration says waste can be delivered. That is a logistics
limit, not a thermal one.

### Files written

| File | Contents |
|---|---|
| `design_map_passive_*.png` | `t_coolers_off`, then `t_geo` per geology |
| `design_map_encapsulation_*.png` | `t_encap` per cooling technology |
| `explore_full_CA_Recycling_Bg-CaF2.csv` | Every design evaluated — nothing is filtered out |
| `run_config.yaml` | The settings that produced all of this |

`run_config.yaml` is worth knowing about. It records the settings actually
used — including the ones the CSVs do not carry, such as the safety factor and
the pre-encapsulation window — and its header gives the command that reproduces
the run. Six months from now that is the difference between a results folder
you can trust and one you cannot identify.

The CSV carries more than the maps show — `h_active`, `T_ambient_active_C`,
peak centreline and surface temperatures, `Q_per_canister_W` and total
`Facility_Duty_W`. Every number is traceable to the conditions that produced
it. Full list in [reference.md](reference.md#output-columns).

---

## Step 8 — Check specific designs

A map tells you the shape of the problem. To decide, you need exact numbers for
designs you are actually considering. Add them to `solver_config.yaml`:

```yaml
candidates:
  - {name: A, radius_m: 0.400, loading_pct: 5}
  - {name: B, radius_m: 0.180, loading_pct: 5}
  - {name: C, radius_m: 0.080, loading_pct: 10}
```

Re-run the same command and each is reported exactly:

```
Name  Radius_m  Loading_Pct  N_canisters Geology Archetype  t_encap_yr  t_coolers_off_yr  t_active_yr  t_geo_yr  Q_per_canister_W  Facility_Duty_W  Min_H_Active
   A     0.400        5.000            1    Salt ForcedAir       0.647             1.100        0.453    18.654         15276.531        15276.531        25.240
   B     0.180        5.000            9    Salt ForcedAir       0.195             0.487        0.292     0.877          5340.160        48061.437        25.139
   C     0.080       10.000           47    Salt ForcedAir       0.097             0.415        0.317     0.834          1595.667        74996.338        25.092
```

Read a row as a timeline. Design **A** is a single large canister: seal it at
7.8 months, run coolers for 5.4 months, then wait — and it cannot be emplaced
for **18.7 years**. Design **B** is nine smaller canisters and reaches the
repository in **0.88 years**, a factor of twenty-one sooner.

That is the trade in one line: **twenty-one times faster to the repository, at
nine times the fleet.** Which is right depends on what interim storage costs
you against what handling and emplacement cost you. The solver will not decide
that.

Two other columns earn their place:

- `Facility_Duty_W` is what the cooling plant must actually remove — 15 kW for
  one canister, 75 kW for forty-seven. Small canisters are easier individually
  and harder collectively.
- `Min_H_Active` is the convective performance you would have to specify if
  none of the named technologies matches your facility. It comes out just above
  `ForcedAir`'s 25 W/(m²·K) here because it is held a degree below the
  temperature limit rather than exactly on it.

Candidates need not lie on the sweep grid — a proposal rarely coincides with a
log-spaced radius.

---

## One thing that will surprise you

Drop `--archetype` so every cooling technology is compared:

```bash
aethon --material CA_Recycling_Bg-CaF2 \
    --radii-min 0.05 --radii-max 0.4 --radii-steps 14 \
    --loadings-min 5 --loadings-max 15 --loadings-steps 6 \
    --repo Salt --output-dir results/tutorial
```

For design A — the single large canister — the three technologies give:

```
 Archetype  h_active  T_ambient_active_C  t_encap_yr  t_active_yr  t_coolers_off_yr
NaturalAir       5.0                40.0    1.078222     0.021535          1.099757
 ForcedAir      25.0                40.0    0.646797     0.452960          1.099757
 WaterPool     750.0                40.0    0.545559     0.554198          1.099757
```

`NaturalAir` — the weakest option, no cooling plant at all — has by far the
*shortest* cooling duration, 0.022 years against 0.55 for a water pool. That is
not a reason to choose it.

Look at the last column: **1.099757 for all three.** `t_coolers_off` does not
depend on the cooling technology; it is a property of the canister and the
waste. A weak technology forces you to wait more than a year before sealing the
waste, by which time it is nearly cool already, so the coolers barely run. A
strong technology lets you seal it after six and a half months and then runs
the coolers for the remainder. The total is identical:

```
t_encap + t_active = t_coolers_off   (constant)
```

The real difference is that the water pool gets the waste into a sealed
canister **six months sooner** — which matters, because until then it sits in
pool or cask storage rather than in a sealed, transportable form.

So compare technologies on **how early they let you seal the waste**, not on
cooling duration. That is exactly what the encapsulation map contours, and why
`t_active` appears in the tables but on no figure. See
[physics.md](physics.md#the-invariant-worth-internalising).

The same reasoning explains why `h` and `ambient_C` always travel together. A
technology at 250 W/(m²·K) in a 25 °C hall can beat one at 750 W/(m²·K) in a
40 °C hall, because convective flux depends on `h·(T_surface − T_ambient)` — a
colder hall substitutes directly for a better coefficient.

---

## Where to go next

- Use your own inventory, material, or geology → [how-to.md](how-to.md)
- What a setting or output column means → [reference.md](reference.md)
- Whether to trust a number → [physics.md](physics.md), especially
  [what the model does not include](physics.md#what-the-model-does-not-include)
