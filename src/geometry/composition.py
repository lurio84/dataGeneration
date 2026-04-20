"""
geometry/composition.py  —  Cargo composition and spec sampling.

Moved from generate_dataset.py (B9 refactor).
"""

import numpy as np

from geometry.meshes import make_primitive_mesh, EUR_D, JACK_FORK_H, JACK_FORK_L


def _spec_w(spec: dict) -> float:
    """X-width of a primitive spec (box→w, cylinder→2r)."""
    return spec.get("w", spec.get("r", 0.0) * 2)


def _spec_d(spec: dict) -> float:
    """Z-depth of a primitive spec (box→d, cylinder→2r)."""
    return spec.get("d", spec.get("r", 0.0) * 2)


def sample_cargo_spec(
    cfg: dict, rng,
    max_w: float = None,
    max_d: float = None,
    max_h: float = None,
) -> dict:
    """Sample a single cargo primitive spec (box or cylinder).

    max_w / max_d: upper-bound clamps on X-width / Z-depth (stacked mode: secondary ≤ primary).
    max_h: upper-bound clamp on height (flat-cargo mode: very low / almost-floor-level).
    """
    flat_min = cfg.get("flat_min_h", 0.03)  # minimum height used only in flat-cargo mode
    if rng.random() < cfg.get("p_cylinder", 0.0):
        # Cylinder footprint is 2r × 2r — respect max_w / max_d so the upper cylinder
        # never exceeds the lower cargo footprint (stability constraint in stacked mode).
        if max_w is not None or max_d is not None:
            max_dim = min(
                max_w if max_w is not None else float("inf"),
                max_d if max_d is not None else float("inf"),
            )
            r_max = min(cfg["cyl_max_r"], max_dim / 2)
        else:
            r_max = cfg["cyl_max_r"]
        r_max = max(cfg["cyl_min_r"], r_max)
        if max_h is not None:
            h_lo, h_hi = flat_min, max(flat_min, min(cfg["cyl_max_h"], max_h))
        else:
            h_lo, h_hi = cfg["cyl_min_h"], cfg["cyl_max_h"]
        return {
            "type": "cylinder",
            "r": float(rng.uniform(cfg["cyl_min_r"], r_max)),
            "h": float(rng.uniform(h_lo, h_hi)),
        }
    w_max = min(cfg["box_max_w"], max_w) if max_w is not None else cfg["box_max_w"]
    d_max = min(cfg["box_max_d"], max_d) if max_d is not None else cfg["box_max_d"]
    if max_h is not None:
        h_lo, h_hi = flat_min, max(flat_min, min(cfg["box_max_h"], max_h))
    else:
        h_lo, h_hi = cfg["box_min_h"], cfg["box_max_h"]
    return {
        "type": "box",
        "w": float(rng.uniform(cfg["box_min_w"], max(cfg["box_min_w"], w_max))),
        "h": float(rng.uniform(h_lo, h_hi)),
        "d": float(rng.uniform(cfg["box_min_d"], max(cfg["box_min_d"], d_max))),
    }


def compose_cargo(
    specs: list,
    mode,           # "stacked" | "tandem" | None
    pallet_top_y: float,
    rng,
) -> tuple:
    """Position 1 or 2 cargo primitives and return positioned meshes.

    Modes
    -----
    stacked : secondary cargo placed on top of primary (Y direction).
    tandem  : secondary cargo placed in front of or behind primary (Z direction).
              Keeps cargo within the pallet footprint — no X-axis spread.

    Returns
    -------
    items : list of (positioned_mesh, placed_spec_dict)
        placed_spec_dict is the original spec augmented with ox/oy/oz keys.
    cargo_back_z : float
        Most negative Z extent across all items — used to anchor the jack.
    """
    if len(specs) == 1 or mode is None:
        spec = specs[0]
        mesh = make_primitive_mesh(spec)
        mesh.translate([0.0, pallet_top_y, 0.0])
        placed = {**spec, "ox": 0.0, "oy": round(pallet_top_y, 4), "oz": 0.0}
        return [(mesh, placed)], -_spec_d(spec) / 2

    spec1, spec2 = specs[0], specs[1]
    mesh1 = make_primitive_mesh(spec1)
    mesh2 = make_primitive_mesh(spec2)

    if mode == "stacked":
        h1 = spec1.get("h", 0.0)
        # Allow small offset only if spec2 is strictly smaller — keeps it on top
        max_ox = max(0.0, (_spec_w(spec1) - _spec_w(spec2)) / 2)
        max_oz = max(0.0, (_spec_d(spec1) - _spec_d(spec2)) / 2)
        ox2 = float(rng.uniform(-max_ox, max_ox))
        oz2 = float(rng.uniform(-max_oz, max_oz))
        oy2 = pallet_top_y + h1
        mesh1.translate([0.0, pallet_top_y, 0.0])
        mesh2.translate([ox2, oy2, oz2])
        # cargo_back_z uses only the BASE item: the stacked item is above, not beside,
        # so it does not affect the jack position (jack goes under the pallet, not the cargo).
        cargo_back_z = -_spec_d(spec1) / 2
        items = [
            (mesh1, {**spec1, "ox": 0.0, "oy": round(pallet_top_y, 4), "oz": 0.0}),
            (mesh2, {**spec2, "ox": round(ox2, 4), "oy": round(oy2, 4),
                     "oz": round(oz2, 4), "placement": "stacked"}),
        ]
        return items, cargo_back_z

    if mode == "tandem":
        # The two items are centred together on the pallet (Z=0):
        #   cargo1 sits in the back half, cargo2 in the front half.
        # Ensemble centre at Z=0 →
        #   oz1 = -(d2 + gap) / 2   (behind centre)
        #   oz2 = +(d1 + gap) / 2   (in front of centre)
        # Constraint for pallet fit: d1 + gap + d2 ≤ EUR_D
        gap = 0.02  # m clearance between items in Z  (matches generate_scene TANDEM_GAP)
        d1, d2 = _spec_d(spec1), _spec_d(spec2)
        oz1 = float(-(d2 + gap) / 2)
        oz2 = float(+(d1 + gap) / 2)
        mesh1.translate([0.0, pallet_top_y, oz1])
        mesh2.translate([0.0, pallet_top_y, oz2])
        # cargo_back_z: back face of cargo1 (most negative Z)
        cargo_back_z = oz1 - d1 / 2   # = -(d1 + d2 + gap) / 2
        items = [
            (mesh1, {**spec1, "ox": 0.0, "oy": round(pallet_top_y, 4), "oz": round(oz1, 4)}),
            (mesh2, {**spec2, "ox": 0.0, "oy": round(pallet_top_y, 4),
                     "oz": round(oz2, 4), "placement": "tandem"}),
        ]
        return items, cargo_back_z

    raise ValueError(f"Unknown compose mode: {mode!r}")


def compose_cargo_on_vehicle(
    spec: dict,
    rng,
    *,
    fork_h: float = JACK_FORK_H,
    fork_l: float = JACK_FORK_L,
) -> tuple:
    """Position a single cargo primitive on top of vehicle forks.

    Vehicle origin is (X=0, Y=0, Z=0) — forks extend in +Z from 0 to fork_l.
    Cargo base sits at Y=fork_h; centre is randomised within the fork footprint.

    fork_h / fork_l default to JACK_FORK_H / JACK_FORK_L for the pallet jack;
    pass FORKLIFT_FORK_H / FORKLIFT_FORK_L when using the carretilla elevadora.

    Returns
    -------
    mesh : positioned TriangleMesh
    placed : spec dict augmented with ox / oy / oz
    """
    d = _spec_d(spec)
    oz_min = d / 2
    oz_max = max(oz_min, fork_l - d / 2)
    oz = float(rng.uniform(oz_min, oz_max))
    ox = float(rng.uniform(-0.20, 0.20))
    mesh = make_primitive_mesh(spec)
    mesh.translate([ox, fork_h, oz])
    placed = {**spec, "ox": round(ox, 4), "oy": round(fork_h, 4), "oz": round(oz, 4)}
    return mesh, placed
