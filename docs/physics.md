# The physics and the model

Why AETHON is built the way it is, and what its numbers do and do not mean.
No commands here — see [tutorial.md](tutorial.md) to run it and
[reference.md](reference.md) to look things up.

If you read only one section, read
[What the model does not include](#what-the-model-does-not-include).

---

## The problem

Vitrified high-level waste keeps producing heat long after the reactor stops.
Fission products and actinides decay, and every decay deposits energy in the
surrounding glass. A freshly loaded canister can generate hundreds of watts per
kilogram of waste.

Borosilicate glass is stable up to roughly 500 °C. Above that it begins to
**devitrify** — crystalline phases nucleate, and the waste form loses the
long-term radionuclide retention that is the whole point of vitrifying it.
Keeping the glass below that limit is a hard requirement for the entire storage
life, not just at emplacement.

Heat has to escape from the centre of the glass to its surface, then from the
surface into whatever surrounds it. Both steps resist, and the balance between
them turns out to drive the entire design problem.

---

## The governing equation

Heat conduction inside the canister, treating it as a long cylinder so that
nothing varies along its length or around its circumference:

```
ρ·Cₚ(T)·∂T/∂t  =  (1/r)·∂/∂r[ r·k(T)·∂T/∂r ]  +  Q(t)
└──── stored ────┘  └──────── conducted ────────┘   └ generated ┘
```

- **Stored** — how fast temperature rises. `ρ·Cₚ` is the energy needed to warm
  a cubic metre by one degree; larger values mean slower response.
- **Conducted** — net heat arriving by conduction. `k` is thermal conductivity.
  The `1/r` and `r` factors are what make this cylindrical: heat flowing
  outward spreads over an ever-larger area, so the geometry itself dilutes the
  flux.
- **Generated** — decay heat, in watts per cubic metre. The source driving
  everything.

Both `k` and `Cₚ` vary with temperature, which makes the equation nonlinear and
forces a numerical solution.

### Boundary conditions

**At the centre (r = 0)** symmetry requires `∂T/∂r = 0`. This is awkward
numerically because the `1/r` term divides by zero there; L'Hôpital's rule
resolves the limit and the centre node picks up a factor of 4.

**At the surface (r = R)** heat arriving by conduction leaves by convection:

```
−k·∂T/∂r = h·(T_surface − T_ambient)
```

a *Robin* condition. The heat transfer coefficient `h` is the single number
describing how good the cooling is — about 5 W/(m²·K) for still air, hundreds
for circulating water.

---

## Decay heat

Specific decay power is represented as a sum of three decaying exponentials:

```
Q(t) = A₁·e^(−λ₁t) + A₂·e^(−λ₂t) + A₃·e^(−λ₃t)      [W/kg], t in years
```

Real waste contains hundreds of nuclides. The preprocessor solves the full
decay chain — including daughters growing in as parents decay — and fits this
three-term approximation. Each term approximates a group of nuclides with
similar half-lives.

Three terms is a deliberate compromise. The form is:

- **Physically motivated** — the groups correspond to short-, medium- and
  long-lived populations.
- **Analytically invertible** — finding when `Q(t)` falls below a limit is a
  simple bisection, which is what makes two of the three milestones cheap.
- **Compact** — six numbers stand in for the whole inventory.

A useful check on any fit: the smallest λ should correspond to a half-life near
**30 years**, because Cs-137 (30.1 yr) and Sr-90 (28.8 yr) dominate the heat
decades out. Nothing in the fitter knows this, so when it emerges the chain
parsing and decay solve are almost certainly correct.

**The single most important property of `Q(t)`: it depends only on time since
shutdown.** Not on the canister, not on the cooling system. That fact does most
of the work in what follows.

---

## Three milestones

The canister's life has three moments that matter, all counted in years from
reactor shutdown:

| Milestone | What happens | What limits it |
|---|---|---|
| `t_encap` | Waste sealed into a canister, placed under active cooling | Peak **centreline** temperature under the chosen cooling technology |
| `t_coolers_off` | Coolers switched off, canister moved to a passive interim store | Steady-state **centreline** temperature under natural convection |
| `t_geo` | Canister emplaced in the geological repository | Centreline **and surface** temperature, with a geology-specific limit |

The cooling plant runs only between the first two:

```
t_active = t_coolers_off − t_encap
```

### Why the interim store exists

The repository imposes a *surface* limit — bentonite clay degrades above about
100 °C, rock salt above about 200 °C. Waiting for the surface to cool takes far
longer than waiting for the centre.

There is no reason to run coolers through that wait. Once the centreline is
passively safe, the canister can sit in an unpowered interim store, where
nothing is in contact with its surface and only the centreline limit applies.
Separating `t_coolers_off` from `t_geo` is often the difference between running
a cooling plant for months and running it for years.

The ordering `t_encap ≤ t_coolers_off ≤ t_geo` holds by construction, not by
assumption: removing the surface constraint can only raise the allowable heat
rate, so the centreline-only milestone cannot fall later than the combined one.

### The invariant worth internalising

Because `Q(t)` depends only on elapsed time, **`t_coolers_off` is a property of
the canister and the waste — not of the cooling technology.**

A better cooling system does not get you to the repository sooner. It lets you
*encapsulate* sooner, and every year earlier you encapsulate is one more year
the coolers must run:

```
t_encap + t_active = t_coolers_off = constant
```

So judging cooling technologies by how little cooling time they need is exactly
backwards — the **weakest** system always wins that comparison, by forcing you
to wait so long before sealing the waste that it is nearly cool already. That
is an artefact of the accounting, not an engineering result.

The meaningful question is how soon a technology lets you get waste into a
sealed canister. That is why the encapsulation map contours `t_encap` and not
`t_active`, and why the two figures are separated the way they are: one shows
what the canister and the waste determine, the other what the facility
determines.

---

## Two solvers, for two different questions

### Transient — "how hot does it get?"

Just after encapsulation the heat output is high and falling fast. Temperature
climbs, peaks, then falls. The peak is what threatens the glass, and finding it
needs the time-dependent equation.

The method is **Method of Lines**: discretise the radius into ~50 points,
replace spatial derivatives with differences between neighbours, and the
partial differential equation becomes 50 coupled ordinary differential
equations in time.

That system is **stiff** — innermost cells respond in seconds while decay heat
changes over years — so an explicit time-stepper would need impractically small
steps. A BDF integrator handles this by solving implicitly at each step.

Integration stops automatically the moment the centre temperature peaks,
detected by watching `dT/dt` at the centre cross zero. There is no value in
continuing.

This solver is used for exactly one gate: `t_encap`. It is the only expensive
part of the calculation.

### Analytical — "when is it safe to leave alone?"

For the passive phases the question is different: not what the peak is, but
whether the canister could sit indefinitely at a given heat output without
overheating. That is a steady-state question with a closed-form answer:

```
T_centre  = T_ambient + Q·R/(2h) + Q·R²/(4k)
T_surface = T_ambient + Q·R/(2h)
```

Those two extra terms are the two resistances heat must overcome, and the
distinction between them is the most useful thing in this document:

- **`Q·R/(2h)`** — getting off the surface into the surroundings. Shrinks as
  cooling improves.
- **`Q·R²/(4k)`** — getting from the centre to the surface through the glass.
  **Does not depend on `h` at all.** It scales with `R²`, so doubling the
  radius quadruples it.

That second term is why large canisters are fundamentally hard: past a certain
size no amount of surface cooling helps, because the bottleneck is internal.
It is also why radius is a decision variable at all.

Inverting these gives the largest survivable heat output:

```
Q_max,centre  = (T_limit,centre  − T_ambient) / [ R/(2h) + R²/(4k) ]
Q_max,surface = (T_limit,surface − T_ambient) / [ R/(2h) ]
Q_allowable   = min of the two
```

Setting the surface limit to infinity makes `Q_max,surface` infinite, so the
minimum selects the centre term alone — which is exactly the interim store,
where nothing touches the canister. The same expression therefore produces both
`t_coolers_off` and `t_geo` depending on the surface limit handed to it.

Using steady state here is deliberately **conservative**: a decaying source
never actually reaches steady state, so real temperatures are always lower than
predicted.

---

## Searching the design space

You choose a canister **radius** and a **waste loading**. Everything else
follows. Three quantities you want small, and cannot minimise together:

- **Number of canisters** — small, lightly loaded canisters run cool but you
  need many more, with the handling, transport and repository footprint that
  implies.
- **Active cooling duration** — how long you pay to run the plant.
- **Time to repository** — how long the waste occupies interim storage.

There is no single best answer, because the weighting between these is a
judgement about cost, risk and logistics, not a calculation.

So the tool does not weigh them. The decision space has only two dimensions —
radius and loading — which is small enough to draw completely, and every
quantity above is a smooth field over that plane. The output is therefore a
**map**, contoured over radius and loading, rather than a ranked list. Nothing
is filtered out; you see the whole space and choose within it.

Repository geology and cooling technology are not optimised over either — you
do not choose your site's rock, and you choose a cooling system rather than a
heat transfer coefficient. They are small discrete sets, enumerated as panels
and compared side by side.

### Why two figures and not eight

The dependency structure does the work:

| Quantity | Varies with | Cost |
|---|---|---|
| `N_canisters` | radius, loading | closed form |
| `t_coolers_off` | radius, loading | analytic root-find |
| `t_geo` | radius, loading, **geology** | analytic root-find |
| `t_encap` | radius, loading, **cooling technology** | transient FEM |

`t_encap` has no geology dependence, and the passive milestones have no
technology dependence. So the passive map needs one panel per geology, the
encapsulation map one panel per technology, and neither needs the product of
the two. The same structure means the two sweeps run independently — nesting
them would solve every transient once per geology and discard all but one.

`t_active` is reported but never plotted. It is not a decision variable:
given a design and a technology it is fully determined, and because
`t_encap + t_active` is constant it collapses toward zero for a weak
technology. Treating it as an objective in its own right ranks the weakest
technology best, which is an accounting artefact rather than a result.

`N_canisters` is not plotted either — it is `ceil` of a closed-form volume
calculation with no physics in it, and it is reported per design in the tables.

Radii are sampled logarithmically, because the `Q·R²/(4k)` term means behaviour
changes fastest at small radii. Both grids are contour axes, so they need
enough points to draw smooth isolines through: a stepped-looking contour means
the grid is too coarse, not that the solver is noisy.

---

## What the model does not include

Every item here is a reason a real canister could behave differently from the
prediction. Read this before quoting a number.

**One spatial dimension.** The canister is treated as an infinite cylinder:
radial conduction only, no variation along its length. End effects and axial
conduction are absent. Reasonable for a long, slender canister; increasingly
wrong for a squat one. Neglecting axial heat loss is conservative — a real
short canister sheds heat through its ends and runs cooler than predicted.

**No canister wall.** The steel wall's thermal resistance is ignored. For
stainless steel (~15 W/(m·K)) at typical wall thicknesses this is a small
correction, but it is not zero.

**One canister at a time.** Neighbouring canisters do not warm each other and
repository spacing plays no part. In a real repository, emplacement density is
often what actually governs the surface temperature — this model cannot tell
you the required spacing.

**Uniform waste distribution.** Heat generation is spread evenly through the
glass. Real waste forms can have composition gradients or hot inclusions.

**Constant properties in the passive phase.** The analytical steady-state
solution evaluates `k` at a fixed temperature rather than resolving its
variation through the canister. The transient solver does not make this
simplification.

**Worst-case values, not distributions.** Ambient temperatures and heat
transfer coefficients are single conservative numbers, not probability
distributions. Results are bounding rather than expected, and the model cannot
express a confidence interval.

**Cooling technology figures are illustrative.** The built-in `(h, ambient)`
pairs are literature-typical convective ranges for orientation, **not vendor
performance data.** Any feasibility conclusion — "forced air is enough",
"you need a chiller" — depends entirely on these numbers and should be re-run
with figures from your own facility design.

**The decay fit has a validity range.** The three-term fit is trustworthy
between the preprocessor's cutoff (default 1 month) and its duration (default
50 years). Below the cutoff it substantially overstates heat; beyond the
duration it extrapolates. Extrapolation is fairly benign — a sum of
exponentials decays toward its longest-lived term, which is physically the
right shape — but a `t_geo` beyond the fitted duration should be treated as
indicative and the fit regenerated.

**Three exponential terms, always.** Decay curves with very sharp early peaks
or unusually long tails may not be captured across their full dynamic range.
The reported R² and RMSE are the check.

---

## Where this is implemented

| Concern | Module |
|---|---|
| Governing equation, transient solve | `aethon/physics/fem_solver.py` |
| Steady-state limits | `aethon/physics/analytical.py` |
| The three milestones | `aethon/analysis/pipeline.py` |
| Geometry and fleet size | `aethon/design/canister.py` |
| Cooling technology library | `aethon/design/archetypes.py` |
| Sweep orchestration | `aethon/design/search.py` |
| Named candidate designs | `aethon/design/candidates.py` |
| Contour maps and console output | `aethon/design/report.py` |
| Decay chain solve and fit | `decay_preprocessor/` |

Material properties are expressions in the config file evaluated at every node
and time step, so changing a material needs no code change.
