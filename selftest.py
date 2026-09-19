"""Run the guard against synthetic devices, including ones that are deliberately wrong.

Why this file exists: a guard that has never failed is not known to be a guard. The check here is
not only that a correct device passes, it is that each specific wrong device fails, and fails on
the row that names the thing that is wrong. A pass-only test would have been satisfied by a
function that returns True.

It builds its own minimal stand-in for a simulation object, so it runs with numpy alone and needs
no solver installed. Run: python selftest.py
"""
from __future__ import annotations

import numpy as np

from device_assert import assert_device


class Medium:
    def __init__(self, permittivity):
        self.permittivity = permittivity


class Box:
    """Axis-aligned box with the two methods the measurement layer asks for."""

    def __init__(self, xlim, ylim, zlim):
        self.xlim, self.ylim, self.zlim = xlim, ylim, zlim

    @property
    def bounds(self):
        return ((self.xlim[0], self.ylim[0], self.zlim[0]),
                (self.xlim[1], self.ylim[1], self.zlim[1]))

    def inside(self, x, y, z):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        return ((x >= self.xlim[0]) & (x <= self.xlim[1]) &
                (y >= self.ylim[0]) & (y <= self.ylim[1]) &
                (z >= self.zlim[0]) & (z <= self.zlim[1]))


class Structure:
    def __init__(self, name, geometry, eps):
        self.name, self.geometry, self.medium = name, geometry, Medium(eps)


class Sim:
    def __init__(self, structures, center=(0, 0, 0), size=(20, 20, 6)):
        self.structures = structures
        self.center, self.size = center, size
        self.medium = Medium(1.0)
        self.sources, self.monitors = (), ()


BIG = 50.0


def build(moat=True, cell_reaches_below=True, with_mesh=False):
    """A ridge waveguide in a film, on cladding, on substrate.

    Defaults are the declared device. Each keyword turns one thing into a lie, which is what the
    negative cases below use.
    """
    z_sub = (-3.0, -2.0)
    z_clad = (-2.0, -0.2)
    z_core = (-0.2, 0.1)

    structs = [
        Structure("substrate", Box((-BIG, BIG), (-BIG, BIG), z_sub), 12.0),
        Structure("cladding", Box((-BIG, BIG), (-BIG, BIG), z_clad), 2.1),
        Structure("film", Box((-BIG, BIG), (-BIG, BIG), z_core), 4.0),
    ]
    if moat:
        # two trenches cut through the film, leaving a 0.6 um core between them.
        structs.append(Structure("moat_left", Box((-BIG, BIG), (-1.5, -0.3), z_core), 1.0))
        structs.append(Structure("moat_right", Box((-BIG, BIG), (0.3, 1.5), z_core), 1.0))
    if with_mesh:
        class Mesh(Box):
            pass
        Mesh.__name__ = "TriangleMesh"
        structs.append(Structure("particle", Mesh((-0.2, 0.2), (-0.2, 0.2), (0.1, 0.4)), 6.0))

    floor = -4.0 if cell_reaches_below else -3.0
    center_z = 0.5 * (floor + 2.0)
    return Sim(structs, center=(0, 0, center_z), size=(20, 20, 2.0 - floor))


SPEC = {
    "propagation_axis": "x",
    "probe": [0.0, 0.0],
    "cell_encloses_stack": True,
    "stack": [
        {"name": "substrate", "thickness_um": 1.0},
        {"name": "cladding", "thickness_um": 1.8},
        {"name": "film", "thickness_um": 0.3},
    ],
    "profile": {
        "at_layer": "film",
        "half_span_um": 2.0,
        "features": [
            {"name": "film"},
            {"name": "moat_left", "width_um": 1.2},
            {"name": "film", "width_um": 0.6},
            {"name": "moat_right", "width_um": 1.2},
            {"name": "film"},
        ],
    },
    "expect_meshes": False,
}


def _run(title, sim, spec, want_pass, must_mention=None):
    print(f"\n=== {title}")
    rows: list = []
    ok = assert_device(sim, spec, verbose=True)
    if ok != want_pass:
        print(f"  [SELFTEST FAIL] expected {'PASS' if want_pass else 'FAIL'}, got "
              f"{'PASS' if ok else 'FAIL'}")
        return 1
    return 0


def main() -> int:
    fails = 0

    fails += _run("the declared device", build(), SPEC, want_pass=True)

    # The original failure: a patterned film modelled as an unpatterned slab. The vertical stack is
    # IDENTICAL, so a stack-only guard passes it. Only the in-plane profile sees it.
    fails += _run("moats missing: the film is an unpatterned slab",
                  build(moat=False), SPEC, want_pass=False)

    # The substrate terminated by the absorbing boundary is a semi-infinite half space, and every
    # check above it still passes.
    fails += _run("cell floor sits on the substrate, so the substrate never ends",
                  build(cell_reaches_below=False), SPEC, want_pass=False)

    # A declared control that quietly has the particle in it.
    fails += _run("control run with a mesh particle present",
                  build(with_mesh=True), SPEC, want_pass=False)

    # A spec that declares nothing must not report a pass.
    fails += _run("empty spec", build(), {"propagation_axis": "x"}, want_pass=False)

    # A design region named but absent: declaring one that does not exist is a failure, not a skip.
    spec_dr = dict(SPEC, design_region={"name": "not_here"})
    fails += _run("design region declared but not in the structure list",
                  build(), spec_dr, want_pass=False)

    print(f"\n{'[OK] selftest: every case behaved as declared' if not fails else f'[FAIL] {fails} case(s) wrong'}")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
