# AETHON

**A**nalysis of **E**ncapsulated **T**hermal **H**eat and **O**ptimised **N**uclides

Works out what cooling infrastructure a vitrified nuclear waste stream needs.
You supply an isotope inventory and the matrix you intend to immobilise it in;
AETHON sweeps canister designs and **maps** when each one can be encapsulated,
when its coolers can stop, and when a repository will accept it.

Nothing is ranked. The output is the design space, not a recommendation.

---

## Documentation

| | |
|---|---|
| **[Tutorial](docs/tutorial.md)** | One worked run, inventory to maps. Start here. |
| **[How-to guides](docs/how-to.md)** | How to use your own inventory, material, geology, cooling technology |
| **[Reference](docs/reference.md)** | Every command, config key, file format and output column |
| **[Physics](docs/physics.md)** | The model, and how it functions |

---

## Install

```bash
git clone <repository-url>
cd aethon
pip install -e .
```

Python 3.10+. No compiled dependencies. Gives you `aethon` and
`decay-preprocessor`. Run them from the repository directory.

---

## Example

```bash
# 1. Fit a decay curve to your isotope inventory
decay-preprocessor \
    --inventory   examples/msr_inventory_5y.csv \
    --chain       data/chain_endfb71_pwr.xml \
    --output-dir  results/decay

# 2. Point solver_config.yaml at the result
#    waste_source: results/decay/waste_source.yaml

# 3. Sweep the design space
aethon --material CA_Recycling_Bg-CaF2 --repo Salt
```

`solver_config.yaml` ships filled in and ready to run. Every setting is
documented in place, and in [reference.md](docs/reference.md#configuration-file).

Decay chain files are ~27 MB and are not bundled; download one from
<https://openmc.org/nuclear-data/>. The [tutorial](docs/tutorial.md) covers
this properly.

You get two maps over canister radius and waste loading:

| Figure | Answers |
|---|---|
| `design_map_passive_*.png` | When can the coolers stop, and when will each repository accept it? |
| `design_map_encapsulation_*.png` | Which canisters can each cooling technology handle, and how soon? |

plus `explore_full_*.csv` with every point evaluated, and `run_config.yaml`
recording the settings that produced them.

### Checking specific designs

Once you have shortlisted candidates, name them in `solver_config.yaml`:

```yaml
candidates:
  - {name: A, radius_m: 0.080, loading_pct: 15}
  - {name: D, radius_m: 0.215, loading_pct: 25}
```

and each run reports them exactly:

```
Name  Radius_m  Loading_Pct  N_canisters Geology Archetype  t_encap_yr  t_coolers_off_yr  t_geo_yr
   A     0.080       15.000           30    Salt ForcedAir       0.192             0.584     1.466
   D     0.215       25.000            1    Salt ForcedAir       1.413            41.110    79.904
```

One canister means waiting 80 years before emplacement; thirty cuts that to
1.5. Which you prefer is yours to decide.

---

## What it models

Three milestones, all measured in **years from reactor shutdown**:

| Milestone | What happens |
|---|---|
| `t_encap` | Waste sealed into a canister, active cooling begins |
| `t_coolers_off` | Centreline passively safe; cooling infrastructure no longer required |
| `t_geo` | Surface passively safe for repository emplacement |



---

## Two things you choose, separately

| You pick | With | It supplies |
|---|---|---|
| The **matrix** — the glass or ceramic block | `material:` | `k(T)`, `cp(T)`, density, devitrification limit |
| The **waste stream** — what is inside it | `waste_source:` | decay curve, campaign mass |

The matrix is the container; the waste stream is the contents. The same waste
can go into different matrices, and the same matrix can hold waste from
different reactors, so they are set independently. ("Waste form" usually means
the combination of the two; AETHON keeps them apart.)

Ships with `BorosilicateGlass` plus two Copenhagen Atomics glass-ceramics.

---

## A caution

The built-in cooling technologies carry **literature-typical convective ranges
for orientation, not vendor performance data**. Any conclusion of the form
"forced air is sufficient" depends entirely on those numbers. Replace them with
figures from your own facility design before relying on the result — see
[how-to.md](docs/how-to.md#define-my-own-cooling-technology).

More generally, read [what the model does not
include](docs/physics.md#what-the-model-does-not-include) before quoting a
number. The model is 1D, single-canister, and worst-case rather than
probabilistic.

---

## Tests

```bash
pytest tests/ -v                # everything
pytest tests/ -v -m "not slow"  # skip the real-chain integration test
```

---

## Licence

See [LICENSE](LICENSE).
