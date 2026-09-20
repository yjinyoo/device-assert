# device-assert

Check that a built electromagnetic simulation is the device you declared, by measuring
the device back out of the simulation object rather than by reading the script that
made it.

```
python selftest.py                              # no solver needed
python device_assert.py my_builder.py device.json
```

```python
from device_assert import assert_device
ok = assert_device(sim, "device.json")
if not ok:
    raise SystemExit("built simulation is not the device")
```

## The failure it exists for

Not a typo. A geometry error that enters as a convenience for one run, and is then
inherited by every run after it, because each new run is checked against the previous
one.

A chain of relative checks does not catch an error, it propagates one. Every run
matches its predecessor, the spectra move smoothly, and the cross-sections look like
last week's because they are. Each run was checked, against the wrong thing.

## How it measures

By point sampling the structure list. For a point, the last structure whose geometry
contains it wins, which is the solver's own overwrite order, so what the guard sees is
what the mesher will see. Reading the builder's variables instead would just re-read
the mistake.

From that it reconstructs two things:

**The vertical stack** on a probe line: which layers, in which order, how thick.

**The in-plane profile** across the propagation axis. A patterned film modelled as an
unpatterned slab has an *identical* vertical stack. The trenches are missing, the mode
is not confined, and every layer check still passes. Only a cut across the device sees
it, as one material running all the way out.

The selftest builds exactly that case and requires it to fail.

## What it checks

Each section of the spec is optional, and what you leave out is not checked. A spec
that declares nothing is reported as a failure rather than as a pass, because an empty
pass is the failure mode this whole module is about.

| section | what it asserts |
| --- | --- |
| `stack` | layers bottom-up on the probe line, by name and thickness |
| `cell_encloses_stack` | the cell extends past the stack, so the outer layer is not terminated by the absorbing boundary |
| `profile` | the in-plane feature sequence and widths across the propagation axis |
| `expect_meshes` | imported mesh structures present, or declared absent |
| `design_region` | a free-form region: measured outside it, optional mirror symmetry and solid corridor |

A bottom layer that runs into the absorbing boundary is not the declared layer, it is
a semi-infinite half space. Every check above it passes, because none of them looks
below the layer it names, and a cross-section plot shows the same coloured band
whether the oxide is two microns thick or never ends.

`expect_meshes` has to be declared in both directions. A run that should carry an
imported particle and does not is a silent null. A control that is supposed to have no
particle and quietly has one destroys the control without failing anything.

## Free-form design regions

A topology-optimised run replaces the drawn film over part of the cell with a
custom medium. Inside that region the structure *name* is constant and says nothing:
it is one structure whose material varies from point to point. A guard that only reads
names degrades to a name lookup at exactly the moment the geometry becomes free.

Declare the region and the guard changes what it does: it measures the stack and the
profile on a cut taken outside the region, where the drawn film survives, and says in
the report which cut it used. It reads permittivity rather than names inside the
region.

The region must be declared, never inferred. A guard that decides for itself which
structure it is allowed to stop measuring can be silenced by naming a structure
conveniently, so declaring a region that does not exist is a failure, not a skip.

## Two measurement details

**An interface is placed at the midpoint between samples.** The obvious version reports
a run from its first sample to its last, which is short by one spacing at each end and
so understates every thickness, always in the same direction. A biased measurement is
worse than a noisy one: it passes a tolerance chosen for float noise until the cell
gets tall enough, and then calls a correct device wrong. Placing the interface between
the last sample of one run and the first of the next leaves an unbiased error of at
most half a spacing.

**A missing dependency is not a geometry that cannot answer.** Testing whether a point
is inside an imported mesh needs a spatial index package. Without it, the containment
call raises, and an except-everything swallows it and reports the surrounding material
instead. Every assertion involving a mesh then passes without ever testing the mesh.
So an import failure is raised, loudly, while a geometry that legitimately cannot
answer is still skipped.

## Requirements

Python 3.9+ and numpy. The measurement layer needs `structures`, each with `.name`,
`.medium` and `.geometry.inside(x, y, z)`, plus `center`, `size` and `medium` on the
simulation object. It was written against Tidy3D; any solver exposing that shape will
work. `selftest.py` supplies its own stand-in, so the guard can be run and judged
without installing a solver.

MIT licensed.
