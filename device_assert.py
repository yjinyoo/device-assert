"""device_assert -- check that a BUILT simulation is the device you declared.

The failure this exists for is not a typo. It is a geometry error that enters as a
convenience for one run and is then inherited by every run after it, because each
new run is checked against the previous one. A chain of relative checks does not
catch an error, it propagates one: every run matches its predecessor, the spectra
move smoothly, and nothing looks wrong for weeks.

So this module does the one thing prose and eyeballing cannot. It reads the BUILT
simulation object, measures the device back out of it, and compares that to a spec
the caller wrote down, with no reference to any earlier run.

    from device_assert import assert_device
    ok = assert_device(sim, "device.json")
    if not ok:
        raise SystemExit("built simulation is not the device")

    python device_assert.py my_builder.py device.json

HOW IT MEASURES. Not by reading the builder's variables, which would only re-read
the mistake, but by point sampling the structure list: for a point, the LAST
structure whose geometry contains it wins, which is Tidy3D's own overwrite order.
From that it reconstructs the vertical layer stack on a probe line and the in-plane
profile across the propagation axis, and compares both to the spec.

Written against Tidy3D. The measurement layer needs `structures`, each with `.name`,
`.medium` and `.geometry.inside(x, y, z)`, plus `center`, `size` and `medium` on the
simulation; any solver exposing that shape will work.
"""
import json
import sys
from pathlib import Path

import numpy as np

TOL_UM = 2e-3          # 2 nm: tighter than any process spec, looser than float noise


# ==================================================================================================
# point sampling: what is at (x, y, z), by Tidy3D's own last-structure-wins order
# ==================================================================================================
def _inside(s, px, py, pz) -> bool:
    """`s.geometry.inside`, with a MISSING DEPENDENCY told apart from a geometry that cannot answer.

    ★ 2026-09-03.  A structure audit found that `rtree` was absent from the environment, so
    `trimesh.contains()` raised ModuleNotFoundError for every TriangleMesh, the three call sites
    below swallowed it, and the guard reported the surrounding host material at a point that is
    inside a mesh particle's air void.  Every device assertion involving a mesh had been passing
    WITHOUT EVER TESTING THE MESH, on every project that imports this module.
    A geometry that legitimately cannot answer is still not evidence and is still skipped.  An
    import failure is not that: it is the machine, it is silent, and it makes a guard read stronger
    than it is.  Same family as the project rule that a cost guard reporting 0.0000 is not a guard.
    """
    try:
        return bool(np.any(s.geometry.inside(px, py, pz)))
    except ImportError as e:               # ModuleNotFoundError is a subclass
        raise RuntimeError(
            f"device_assert cannot test the geometry of {s.name!r}: {e}.  This is an ENVIRONMENT "
            f"failure, not a geometry one -- the guard would otherwise pass without looking at "
            f"this structure.  Install the missing package (a TriangleMesh needs `rtree`) and "
            f"re-run.") from e
    except Exception:
        return False                       # a geometry that cannot answer is not evidence


def structure_at(sim, x: float, y: float, z: float, skip: tuple[str, ...] = ()) -> str | None:
    """Name of the last structure containing the point, or None for the background medium.

    skip exists for TriangleMesh structures: inside() on a mesh is slow, and this module never
    needs to sample inside one, only to confirm it is present and where its bounds are.
    """
    hit = None
    px, py, pz = (np.array([v], dtype=float) for v in (x, y, z))
    for s in sim.structures:
        if s.name in skip:
            continue
        if _inside(s, px, py, pz):
            hit = s.name
    return hit


def _mesh_names(sim) -> tuple[str, ...]:
    """Names of the imported-mesh structures, which are skipped when point sampling.

    The isinstance test is the reliable one, but it must not be the only one: importing the solver
    just to classify a geometry would make this module unusable without it, and a guard nobody can
    run is a guard that gets deleted. So the type name is the fallback, and the import is optional.
    """
    mesh_types: tuple = ()
    try:
        import tidy3d as td
        mesh_types = (td.TriangleMesh,)
    except Exception:
        pass
    out = []
    for s in sim.structures:
        g = s.geometry
        if (mesh_types and isinstance(g, mesh_types)) or \
                type(g).__name__ in ("TriangleMesh", "Transformed"):
            out.append(s.name)
    return tuple(out)


def _runs_from_samples(coords, names):
    """[(name, from, to), ...] from a sampled line, with each interface at the MIDPOINT.

    The obvious version reports a run from its first sample to its last, which is short by one
    sample spacing at each end and therefore UNDERSTATES every thickness, always in the same
    direction. A biased measurement is worse than a noisy one here: it survives a tolerance chosen
    for float noise until the cell gets tall enough, and then reports a device that is right as
    wrong. The true interface lies between the last sample of one run and the first of the next, so
    that is where it is placed, which leaves an unbiased error of at most half a spacing.

    The outer edges stay at the scan limits, because outside the scan nothing was measured.
    """
    n = len(coords)
    runs = []
    start = 0
    for i in range(1, n + 1):
        if i == n or names[i] != names[start]:
            lo = float(coords[0]) if start == 0 else float(0.5 * (coords[start - 1] + coords[start]))
            hi = (float(coords[n - 1]) if i == n
                  else float(0.5 * (coords[i - 1] + coords[i])))
            runs.append((names[start], lo, hi))
            start = i
    return runs


def stack_along_z(sim, x: float, y: float, z_lo: float, z_hi: float, n: int = 4000):
    """[(name, z_bottom, z_top), ...] bottom-up on the line (x, y), from point sampling."""
    skip = _mesh_names(sim)
    zs = np.linspace(z_lo, z_hi, n)
    names = [structure_at(sim, x, y, float(z), skip=skip) for z in zs]
    return _runs_from_samples(zs, names)


def profile_across(sim, z: float, axis: str, half_span: float, n: int = 4000,
                   at: float = 0.0):
    """[(name, t_from, t_to), ...] across the waveguide at height z, from point sampling.

    axis is the PROPAGATION axis, so the scan runs along the other in-plane axis.
    `at` is the coordinate ALONG the propagation axis to cut at; it is 0 for a drawn device and is
    moved outside the design region when one is declared.
    """
    skip = _mesh_names(sim)
    ts = np.linspace(-half_span, half_span, n)
    if axis == "x":
        names = [structure_at(sim, at, float(t), z, skip=skip) for t in ts]
    else:
        names = [structure_at(sim, float(t), at, z, skip=skip) for t in ts]
    return _runs_from_samples(ts, names)


# ==================================================================================================
# permittivity sampling: needed because inside a design region the STRUCTURE NAME is constant and
# says nothing. A free-form region is one structure whose material varies from point to point, so a
# guard that only reads names cannot see whether the guide survived inside it.
# ==================================================================================================
def _eps_of_medium(med, x: float, y: float, z: float):
    """Relative permittivity of a medium at a point, or None if it cannot be determined."""
    perm = getattr(med, "permittivity", None)
    if perm is None:
        for comp in ("xx", "yy", "zz"):                       # anisotropic
            sub = getattr(med, comp, None)
            if sub is not None:
                return _eps_of_medium(sub, x, y, z)
        return None
    if hasattr(perm, "sel"):                                  # CustomMedium: a SpatialDataArray
        try:                                                  # nearest, never extrapolate
            return float(perm.sel(x=x, y=y, z=z, method="nearest").values)
        except Exception:
            return None
    try:
        return float(perm)
    except (TypeError, ValueError):
        return None


def eps_at(sim, x: float, y: float, z: float, skip: tuple[str, ...] = ()):
    """Relative permittivity at a point, by the same last-structure-wins order as structure_at."""
    hit = None
    px, py, pz = (np.array([v], dtype=float) for v in (x, y, z))
    for s in sim.structures:
        if s.name in skip:
            continue
        if _inside(s, px, py, pz):
            hit = s
    return _eps_of_medium(hit.medium if hit is not None else sim.medium, x, y, z)


def _index_at(sim, x: float, y: float, z: float):
    """Refractive index at a point, evaluated at the simulation's own source frequency.

    eps_at() reads `medium.permittivity`, which a dispersive library material does not carry, so it
    returns None for silicon.  This asks the medium to model itself instead.  (2026-08-18)
    """
    eps = eps_at(sim, x, y, z)
    if eps is not None:
        try:
            return float(np.sqrt(complex(eps).real))
        except (TypeError, ValueError):
            pass
    freq = None
    for src in getattr(sim, "sources", ()):
        f0 = getattr(getattr(src, "source_time", None), "freq0", None)
        if f0:
            freq = float(f0)
            break
    if freq is None:                       # cross-section builds carry no source
        for mon in getattr(sim, "monitors", ()):
            fs = getattr(mon, "freqs", None)
            if fs is not None and len(fs):
                freq = float(fs[len(fs) // 2])
                break
    px, py, pz = (np.array([v], dtype=float) for v in (x, y, z))
    hit = None
    for s in sim.structures:
        if _inside(s, px, py, pz):
            hit = s
    med = hit.medium if hit is not None else sim.medium
    if freq is None:                       # neither source nor monitor: ask the medium itself
        fr = getattr(med, "frequency_range", None)
        if fr:
            freq = 0.5 * (float(fr[0]) + float(fr[1]))
    if freq is None:
        return None
    try:
        return float(np.sqrt(complex(med.eps_model(freq)).real))
    except Exception:
        return None


def _find_structure(sim, name: str):
    for s in sim.structures:
        if s.name == name:
            return s
    return None


def _probe_outside(sim, struct, axis: str, margin: float = 0.30):
    """A coordinate along the propagation axis that is OUTSIDE the design region and inside the cell.

    Returns (coord, description) or (None, why not). Preference goes to the side with more room, so
    the cut lands in undisturbed drawn film rather than just barely clear of the region.
    """
    i = 0 if axis == "x" else 1
    (lo_s, hi_s) = struct.geometry.bounds[0][i], struct.geometry.bounds[1][i]
    lo_c = sim.center[i] - sim.size[i] / 2 + margin
    hi_c = sim.center[i] + sim.size[i] / 2 - margin
    gaps = []
    if hi_c - hi_s > 2 * margin:
        gaps.append((hi_c - hi_s, 0.5 * (hi_s + hi_c), f"+{axis} side"))
    if lo_s - lo_c > 2 * margin:
        gaps.append((lo_s - lo_c, 0.5 * (lo_c + lo_s), f"-{axis} side"))
    if not gaps:
        return None, (f"the design region spans {lo_s:+.2f} to {hi_s:+.2f} um and leaves no room "
                      f"inside the cell to cut the drawn device")
    gaps.sort(reverse=True)
    _, coord, side = gaps[0]
    return coord, f"{axis} = {coord:+.3f} um, {side} of the region"


def _report(rows, verbose: bool) -> bool:
    ok = all(r[1] for r in rows)
    if verbose:
        width = max(len(r[0]) for r in rows) if rows else 10
        print("  device_assert:")
        for name, good, detail in rows:
            tag = "PASS" if good else "FAIL"
            print(f"    [{tag}] {name:<{width}}  {detail}")
        print(f"  -> {'DEVICE OK' if ok else 'NOT THE DEVICE'}")
    return ok




# ==================================================================================================
# the declarative layer: compare the measured device against a spec the caller wrote down
# ==================================================================================================
def _runs_to_layers(runs, z_lo, z_hi):
    """Drop the runs that are just the cell's background above and below the stack."""
    keep = [r for r in runs if r[0] is not None]
    return keep


def _fmt_um(v):
    return f"{v * 1e3:.0f} nm" if abs(v) < 1 else f"{v:.3f} um"


def _match_name(pattern, name):
    """A layer matches if the declared substring is in the structure's name, case-insensitively.

    Substring rather than equality because builders decorate names ('core', 'core_left'), and
    equality would fail on a device that is right. The spec author picks how specific to be, and a
    pattern that matches two different layers is caught by the ORDER check below, not here.
    """
    if name is None:
        return False
    return str(pattern).lower() in str(name).lower()


def check_stack(sim, spec, rows, probe, axis_note=""):
    """Measure the vertical stack on the probe line and compare it to spec['stack'].

    The spec lists layers bottom-up. Each entry is {"name": <substring>, "thickness_um": <float>},
    and thickness may be omitted for a layer whose extent is set by the cell rather than by design
    (a substrate, a cladding that runs to the boundary).
    """
    layers = spec.get("stack")
    if not layers:
        return None
    px, py = probe
    z_lo = sim.center[2] - sim.size[2] / 2
    z_hi = sim.center[2] + sim.size[2] / 2
    runs = _runs_to_layers(stack_along_z(sim, px, py, z_lo, z_hi), z_lo, z_hi)

    chk = _checker(rows)
    found = [r[0] for r in runs]
    chk(f"stack has {len(layers)} named layer(s) on the probe line{axis_note}",
        len(runs) == len(layers),
        f"measured bottom-up: {found}")
    if len(runs) != len(layers):
        return runs

    for want, got in zip(layers, runs):
        name, z0, z1 = got
        ok_name = _match_name(want["name"], name)
        chk(f"layer {want['name']!r} is where the spec puts it", ok_name,
            f"measured {name!r} from {z0:+.3f} to {z1:+.3f} um")
        t_want = want.get("thickness_um")
        if t_want is None:
            continue
        t_got = z1 - z0
        tol = float(want.get("tol_um", TOL_UM))
        chk(f"layer {want['name']!r} is {_fmt_um(float(t_want))} thick",
            abs(t_got - float(t_want)) <= tol,
            f"measured {_fmt_um(t_got)}, spec {_fmt_um(float(t_want))}, tol {_fmt_um(tol)}")
    return runs


def check_cell_encloses_stack(sim, spec, runs, rows):
    """The cell must extend past the lowest declared layer, or that layer never ends.

    A stack whose bottom layer runs into the absorbing boundary is not the declared stack: it is a
    semi-infinite half space. Every check above it can pass on that geometry, because none of them
    looks below the layer they name. The failure is invisible in a cross-section plot too, which
    shows the same coloured band either way.
    """
    if not spec.get("cell_encloses_stack") or not runs:
        return
    chk = _checker(rows)
    floor = sim.center[2] - sim.size[2] / 2
    ceil_ = sim.center[2] + sim.size[2] / 2
    z_bot = min(r[1] for r in runs)
    z_top = max(r[2] for r in runs)
    below = z_bot - floor
    above = ceil_ - z_top
    chk("the cell reaches below the bottom of the stack",
        below > TOL_UM,
        f"cell floor {floor:+.3f} um, stack bottom {z_bot:+.3f} um"
        + ("" if below > TOL_UM else "   <-- the bottom layer is terminated by the boundary: "
                                     "it never ends"))
    chk("the cell reaches above the top of the stack",
        above > TOL_UM,
        f"cell ceiling {ceil_:+.3f} um, stack top {z_top:+.3f} um")


def check_profile(sim, spec, rows, cut_at=0.0, axis_note=""):
    """Measure the in-plane cut across the propagation axis and compare it to spec['profile'].

    This is the check that catches a patterned film modelled as an unpatterned slab. A missing
    trench, a missing waveguide, a film that was supposed to be interrupted and is not, all show up
    here as one material running all the way out, and nowhere else: the vertical stack is identical
    either way, and so is every spectrum until the mode leaks.

    spec['profile'] = {
        "at_layer": <substring of the layer to cut through, measured at its mid-height>,
        "half_span_um": <how far out to scan>,
        "features": [{"name": <substring or null for background>, "width_um": <float>}, ...]
    }
    Features are listed left to right across the axis. `null` means the background medium, which is
    how a trench or an air moat is declared.
    """
    prof = spec.get("profile")
    if not prof:
        return
    chk = _checker(rows)
    axis = spec.get("propagation_axis", "x")

    z_cut = _layer_mid(sim, spec, prof.get("at_layer"))
    if z_cut is None:
        chk("profile: the layer to cut through was found", False,
            f"no layer matching {prof.get('at_layer')!r} on the probe line")
        return

    runs = [r for r in profile_across(sim, z_cut, axis, float(prof["half_span_um"]), at=cut_at)]
    want = prof["features"]

    # Trim the runs that are only the scan overshooting past the declared features.
    measured = [(n, a, b) for (n, a, b) in runs]
    chk(f"profile across {axis} has {len(want)} feature(s) at z = {z_cut:+.3f} um{axis_note}",
        len(measured) == len(want),
        f"measured: {[(n, round(b - a, 3)) for n, a, b in measured]}")
    if len(measured) != len(want):
        return

    for w, (name, t0, t1) in zip(want, measured):
        wn = w.get("name")
        ok_name = (name is None) if wn is None else _match_name(wn, name)
        chk(f"feature {wn if wn is not None else 'background'!r} in the expected order", ok_name,
            f"measured {name!r} from {t0:+.3f} to {t1:+.3f} um")
        w_want = w.get("width_um")
        if w_want is None:
            continue
        w_got = t1 - t0
        tol = float(w.get("tol_um", 2 * TOL_UM))
        chk(f"feature {wn if wn is not None else 'background'!r} is {_fmt_um(float(w_want))} wide",
            abs(w_got - float(w_want)) <= tol,
            f"measured {_fmt_um(w_got)}, spec {_fmt_um(float(w_want))}, tol {_fmt_um(tol)}")


def check_meshes(sim, spec, rows):
    """Declared presence or absence of TriangleMesh structures.

    Both directions matter. A run that should carry an imported particle and does not is a silent
    null; a CONTROL that is supposed to have no particle, and quietly has one, destroys the control
    without failing anything. The spec has to say which it is, so neither can pass by default.
    """
    if "expect_meshes" not in spec:
        return
    chk = _checker(rows)
    want = spec["expect_meshes"]
    meshes = _mesh_names(sim)
    if want is True:
        chk("an imported mesh structure is present", len(meshes) >= 1,
            f"mesh structures: {list(meshes)}")
    elif want is False:
        chk("no mesh structure is present (declared control)", len(meshes) == 0,
            f"mesh structures: {list(meshes)}")
    else:
        chk(f"exactly {int(want)} mesh structure(s) present", len(meshes) == int(want),
            f"mesh structures: {list(meshes)}")


def check_design_region(sim, spec, rows):
    """Checks for a free-form (topology-optimised) region, which must be DECLARED, never inferred.

    Inside such a region the structure NAME is constant and says nothing: it is one structure whose
    material varies point to point. A guard that only reads names therefore degrades to a name
    lookup at exactly the moment the geometry becomes free. Permittivity has to be read instead.

    A guard that guesses which structure it may stop measuring is a guard that can be silenced by
    naming a structure conveniently, so declaring a region that does not exist is a FAIL.

    spec['design_region'] = {
        "name": <structure name, exact>,
        "mirror_symmetric_about_axis": <bool>,   # a control's attribution can depend on this
        "corridor_um": <float or null>           # a corridor that must stay solid end to end
    }
    """
    dr = spec.get("design_region")
    if not dr:
        return None
    chk = _checker(rows)
    axis = spec.get("propagation_axis", "x")
    struct = _find_structure(sim, dr["name"])
    chk(f"declared design region {dr['name']!r} exists", struct is not None,
        "declared but not found in the structure list" if struct is None else "found")
    if struct is None:
        return None

    coord, why = _probe_outside(sim, struct, axis)
    chk("a cut outside the design region is available, where the drawn film survives",
        coord is not None, why)

    if dr.get("mirror_symmetric_about_axis"):
        _check_region_symmetry(sim, spec, struct, rows)
    if dr.get("corridor_um"):
        _check_corridor(sim, spec, struct, float(dr["corridor_um"]), rows)
    return coord


def _check_region_symmetry(sim, spec, struct, rows, n=81):
    """Permittivity inside the region must mirror about the propagation axis.

    Not an aesthetic constraint. Where a result is attributed to one element by showing that a
    control without it gives zero, a region that is itself asymmetric destroys the control: the
    asymmetry can then come from the region, and the attribution no longer holds.
    """
    chk = _checker(rows)
    axis = spec.get("propagation_axis", "x")
    i = 0 if axis == "x" else 1
    b0, b1 = struct.geometry.bounds
    lo, hi = b0[i], b1[i]
    j = 1 - i
    half = min(abs(b0[j]), abs(b1[j]))
    z = 0.5 * (b0[2] + b1[2])
    worst, worst_at = 0.0, None
    for s in np.linspace(lo + 1e-3, hi - 1e-3, 9):
        for t in np.linspace(0.05 * half, 0.95 * half, n // 9):
            p = (float(s), float(t)) if i == 0 else (float(t), float(s))
            q = (float(s), float(-t)) if i == 0 else (float(-t), float(s))
            e1 = eps_at(sim, p[0], p[1], z)
            e2 = eps_at(sim, q[0], q[1], z)
            if e1 is None or e2 is None:
                continue
            d = abs(e1 - e2)
            if d > worst:
                worst, worst_at = d, (p, e1, e2)
    chk("design region is mirror-symmetric about the propagation axis",
        worst < 1e-6,
        "symmetric to within 1e-6" if worst_at is None or worst < 1e-6 else
        f"worst mismatch {worst:.3g} at {worst_at[0]}: eps {worst_at[1]:.4f} vs {worst_at[2]:.4f}")


def _check_corridor(sim, spec, struct, width_um, rows, n=200):
    """A corridor of the given width along the axis must stay one solid material through the region.

    Declared by the caller, because whether the optimiser is allowed to interrupt the guide is a
    design decision and not something the geometry can answer.
    """
    chk = _checker(rows)
    axis = spec.get("propagation_axis", "x")
    i = 0 if axis == "x" else 1
    b0, b1 = struct.geometry.bounds
    z = 0.5 * (b0[2] + b1[2])
    ref = None
    breaks = []
    for s in np.linspace(b0[i] + 1e-3, b1[i] - 1e-3, n):
        for t in (-width_um / 2, 0.0, width_um / 2):
            p = (float(s), float(t)) if i == 0 else (float(t), float(s))
            e = eps_at(sim, p[0], p[1], z)
            if e is None:
                breaks.append((round(float(s), 3), "no medium"))
                continue
            if ref is None:
                ref = e
            elif abs(e - ref) > 1e-3 * max(1.0, abs(ref)):
                breaks.append((round(float(s), 3), round(e, 4)))
    chk(f"a {_fmt_um(width_um)} corridor stays solid through the region",
        not breaks,
        "solid end to end" if not breaks else
        f"{len(breaks)} sample(s) differ from eps {ref:.4f}, first at {axis} = {breaks[0][0]}")


def _layer_mid(sim, spec, name_pattern):
    """Mid-height of the named layer on the probe line, or None."""
    if name_pattern is None:
        return None
    probe = spec.get("probe", [0.0, 0.0])
    z_lo = sim.center[2] - sim.size[2] / 2
    z_hi = sim.center[2] + sim.size[2] / 2
    for name, z0, z1 in stack_along_z(sim, probe[0], probe[1], z_lo, z_hi):
        if _match_name(name_pattern, name):
            return 0.5 * (z0 + z1)
    return None


def _checker(rows):
    def chk(label, good, detail=""):
        rows.append((label, bool(good), detail))
    return chk


def assert_device(sim, spec, verbose: bool = True) -> bool:
    """Measure the device out of a BUILT simulation and compare it to a declared spec.

    Returns True only if every item passes, and prints a per-item table when verbose.

    `spec` is a dict or a path to a JSON file. Nothing is inferred from the builder: every
    expectation is something the caller wrote down, and every measurement is taken from the
    structure list of the simulation object that is about to be submitted.

    See `device.example.json` for the full schema and `README.md` for why it is shaped this way.
    """
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text(encoding="utf-8"))
    rows: list = []

    cut_at = check_design_region(sim, spec, rows)
    axis_note = ""
    if spec.get("design_region"):
        if cut_at is None:
            return _report(rows, verbose)      # nowhere safe to measure: do not pretend otherwise
        axis_note = f" (cut outside the design region, at {cut_at:+.3f} um)"
    else:
        cut_at = 0.0

    probe = spec.get("probe")
    if probe is None:
        axis = spec.get("propagation_axis", "x")
        probe = [cut_at, 0.0] if axis == "x" else [0.0, cut_at]

    runs = check_stack(sim, spec, rows, probe, axis_note)
    check_cell_encloses_stack(sim, spec, runs, rows)
    check_profile(sim, spec, rows, cut_at=cut_at, axis_note=axis_note)
    check_meshes(sim, spec, rows)

    if not rows:
        rows.append(("the spec declares at least one check", False,
                     "spec has no stack, profile, mesh or design-region section: nothing was "
                     "checked, and an empty pass is not a pass"))
    return _report(rows, verbose)


def main() -> int:
    import argparse
    import importlib.util

    ap = argparse.ArgumentParser(
        description="Check that a built Tidy3D simulation is the device a spec declares.")
    ap.add_argument("builder", help="python file that builds the simulation")
    ap.add_argument("spec", help="device spec JSON")
    ap.add_argument("--attr", default="build_sim",
                    help="callable in the builder that returns the Simulation (default: build_sim)")
    args = ap.parse_args()

    path = Path(args.builder).resolve()
    s = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(s)
    sys.path.insert(0, str(path.parent))
    s.loader.exec_module(mod)
    sim = getattr(mod, args.attr)()
    ok = assert_device(sim, args.spec)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
