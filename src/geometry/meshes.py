"""
geometry/meshes.py  —  Mesh construction helpers.

Moved from generate_dataset.py (B9 refactor).
"""

from pathlib import Path

import numpy as np
import open3d as o3d

# EUR pallet dimensions (m)
EUR_W, EUR_H, EUR_D = 1.20, 0.144, 0.80

# Pallet jack geometry constants — exported so composition/placement logic
# can use them without duplicating magic numbers.
JACK_FORK_H:  float = 0.06   # top surface of forks (platform height for cargo)
JACK_FORK_L:  float = 1.15   # fork length in +Z direction
JACK_BODY_D:  float = 0.40   # body depth in -Z direction
JACK_X_HALF:  float = 0.35   # half-width of full vehicle bounding box

# Carretilla elevadora (forklift) geometry constants.
# Derived from data/carretilla.stl:
#   STL units: mm.  BBox (mm): X=[−0,1270], Y=[0,2785], Z=[−95,3978].
#   After scale=0.001 and placement (forks assumed at Z_max, fork root → world Z=0):
#     X: [−0.635, +0.635]  (width 1.27 m, centred)
#     Y: [0, 2.785]        (height; Y_min=0 already sits on floor)
#     Z: [−2.97, +1.10]    (body in −Z, forks in +Z)
# FORKLIFT_FORK_L / FORKLIFT_FORK_H are physical estimates — verify visually.
FORKLIFT_FORK_H:  float = 0.226  # fork top surface height measured from carretilla.stl (Y p99=0.226m)
FORKLIFT_FORK_L:  float = 1.10   # estimated fork length in +Z direction (m)
FORKLIFT_BODY_D:  float = 2.97   # body depth in −Z: total_len(4.07) − FORK_L(1.10)
FORKLIFT_X_HALF:  float = 0.635  # half-width (STL X extent 1.270 m, centred at 0)


def make_pallet_mesh() -> o3d.geometry.TriangleMesh:
    """EUR pallet with realistic top-deck slat geometry.

    5 boards (1200 × 22 mm) run along X, separated by 19 mm gaps in Z.
    A solid lower support box fills the remaining height beneath the boards.
    Overall bounding box is identical to the old solid version:
      X: [-EUR_W/2, EUR_W/2],  Y: [0, EUR_H],  Z: [-EUR_D/2, EUR_D/2]

    The inter-slat gaps let floor points appear through the pallet top deck,
    creating distinctive height_range_local and curvature features that help
    the classifier separate pallet from flat floor.
    """
    slat_h = 0.022          # board thickness (22 mm)
    n_slats = 5
    gap = 0.019             # gap between boards (19 mm)
    d_slat = (EUR_D - (n_slats - 1) * gap) / n_slats   # ≈ 0.1448 m

    parts: list[o3d.geometry.TriangleMesh] = []

    # Lower support structure — solid box beneath the boards
    support_h = EUR_H - slat_h
    support = o3d.geometry.TriangleMesh.create_box(EUR_W, support_h, EUR_D)
    support.translate([-EUR_W / 2, 0.0, -EUR_D / 2])
    parts.append(support)

    # Top deck: 5 boards running along X (full width), spaced in Z
    board_y = EUR_H - slat_h    # bottom face of top boards
    for i in range(n_slats):
        z0 = -EUR_D / 2 + i * (d_slat + gap)
        board = o3d.geometry.TriangleMesh.create_box(EUR_W, slat_h, d_slat)
        board.translate([-EUR_W / 2, board_y, z0])
        parts.append(board)

    mesh = parts[0]
    for p in parts[1:]:
        mesh = mesh + p
    return mesh


def make_box_mesh(w: float, h: float, d: float) -> o3d.geometry.TriangleMesh:
    """Axis-aligned box centred in X and Z, bottom face at Y=0."""
    m = o3d.geometry.TriangleMesh.create_box(w, h, d)
    m.translate([-w / 2, 0.0, -d / 2])
    return m


def make_cylinder_mesh(r: float, h: float) -> o3d.geometry.TriangleMesh:
    """Upright cylinder centred in XZ, bottom face at Y=0.
    Open3D create_cylinder is Z-aligned by default → rotate 90° around X to make it Y-aligned.
    """
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=h, resolution=32)
    R = np.array([[1, 0,  0],
                  [0, 0, -1],
                  [0, 1,  0]], dtype=np.float64)
    m.rotate(R, center=(0.0, 0.0, 0.0))
    # After rotation cylinder spans Y=-h/2..+h/2 → translate up so bottom sits on Y=0
    m.translate([0.0, h / 2, 0.0])
    return m


def make_primitive_mesh(spec: dict) -> o3d.geometry.TriangleMesh:
    """Create mesh from a cargo spec dict. Bottom at Y=0, centred in XZ."""
    t = spec["type"]
    if t == "box":
        return make_box_mesh(spec["w"], spec["h"], spec["d"])
    if t == "cylinder":
        return make_cylinder_mesh(spec["r"], spec["h"])
    raise ValueError(f"Unknown primitive type: {t!r}")


def make_person_mesh(
    height: float = 1.75,
    y_rotation_deg: float = 0.0,
    stl_path: "str | Path | None" = None,
) -> o3d.geometry.TriangleMesh:
    """Person mesh: intenta cargar data/person.stl; si no existe, usa cilindro+esfera.

    Args:
        height:         Altura objetivo en metros (base en Y=0, cima en Y≈height).
        y_rotation_deg: Rotación aleatoria alrededor del eje Y (0-360°).
        stl_path:       Ruta explícita al STL (tests); None → detecta automáticamente.

    Invariantes de salida:
        vertices[:,1].min() ≈ 0.0        (base en suelo)
        vertices[:,1].max() ≈ height      (cima a la altura pedida)
        centrado en XZ alrededor de X=0, Z=0
    """
    _stl = Path(stl_path) if stl_path is not None else Path(__file__).parent.parent.parent / "data" / "person.stl"

    if _stl.exists():
        mesh = o3d.io.read_triangle_mesh(str(_stl))
        if len(mesh.vertices) == 0:
            raise RuntimeError(f"person.stl cargado vacío: {_stl}")
        mesh.compute_vertex_normals()
        # Escalar a altura objetivo (invariante a las unidades del STL)
        verts = np.asarray(mesh.vertices)
        current_h = verts[:, 1].max() - verts[:, 1].min()
        if current_h > 1e-6:
            mesh.scale(height / current_h, center=(0.0, 0.0, 0.0))
        # Base en Y=0, centrado en XZ
        verts = np.asarray(mesh.vertices)
        cx = (verts[:, 0].max() + verts[:, 0].min()) / 2.0
        cz = (verts[:, 2].max() + verts[:, 2].min()) / 2.0
        mesh.translate([-cx, -verts[:, 1].min(), -cz])
    else:
        # Fallback: cilindro + esfera, proporcionales a height.
        # Altura total original: body_h=0.95 + head_r*2=0.28 → ~1.23m si head centrado a body_h+head_r.
        # Repartimos: 84% cuerpo, 16% cabeza (radio=8%).
        body_h = height * 0.84
        head_r = height * 0.08
        body_r = 0.18
        body = o3d.geometry.TriangleMesh.create_cylinder(radius=body_r, height=body_h, resolution=16)
        # Open3D crea el cilindro alineado con Z → rotamos 90° en X para que quede en Y
        R = np.array([[1, 0, 0],
                      [0, 0, -1],
                      [0, 1,  0]], dtype=np.float64)
        body.rotate(R, center=(0.0, 0.0, 0.0))
        body.translate([0.0, body_h / 2.0, 0.0])   # base en Y=0
        head = o3d.geometry.TriangleMesh.create_sphere(radius=head_r, resolution=8)
        head.translate([0.0, body_h + head_r, 0.0])
        mesh = body + head

    # Rotación Y aleatoria (orientación de la persona en el plano XZ)
    a = np.deg2rad(y_rotation_deg)
    R_y = np.array([
        [ np.cos(a), 0.0, np.sin(a)],
        [       0.0, 1.0,       0.0],
        [-np.sin(a), 0.0, np.cos(a)],
    ], dtype=np.float64)
    mesh.rotate(R_y, center=(0.0, 0.0, 0.0))
    return mesh


def make_pallet_jack_mesh() -> o3d.geometry.TriangleMesh:
    """Manual pallet jack (transpaleta manual).

    Origin = fork-root / body-front at (X=0, Y=0, Z=0).
    Forks extend in +Z (toward cargo/pallet).
    Body and tiller extend in -Z (operator side).

    Components:
      Body  (pump unit):  X:-0.35..+0.35,  Y:0..0.15,   Z:-0.40..0
      Fork L (horca izq): X:-0.30..-0.16,  Y:0..0.06,   Z:0..+1.15
      Fork R (horca der): X:+0.16..+0.30,  Y:0..0.06,   Z:0..+1.15
      Tiller arm L:       X:-0.17..-0.13,  Y:0.15..0.90, Z:-0.22..-0.18
      Tiller arm R:       X:+0.13..+0.17,  Y:0.15..0.90, Z:-0.22..-0.18
      Tiller crossbar:    X:-0.17..+0.17,  Y:0.86..0.90, Z:-0.22..-0.18

    Overall bounding box: X:[-0.35,+0.35] (width=0.70), Y:[0,0.90], Z:[-0.40,+1.15]
    Same XZ footprint as before so generate_dataset.py placement constants are unchanged.

    The U-shaped tiller creates a distinctive tall vertical+horizontal structure
    with high `linearity` and `lam_ratio_12` features — separating vehicle from
    the compact cargo class geometrically rather than positionally.
    """
    # Body: low compact pump unit
    body = o3d.geometry.TriangleMesh.create_box(0.70, 0.15, 0.40)
    body.translate([-0.35, 0.0, -0.40])

    # Forks: flat, elongated — extend in +Z toward cargo/pallet
    fork_l = o3d.geometry.TriangleMesh.create_box(0.14, 0.06, JACK_FORK_L)
    fork_l.translate([-0.30, 0.0, 0.0])
    fork_r = o3d.geometry.TriangleMesh.create_box(0.14, 0.06, JACK_FORK_L)
    fork_r.translate([ 0.16, 0.0, 0.0])

    # Tiller (timón): U-shaped handle rising from body back
    # Two vertical arms + one horizontal crossbar at top
    arm_w, arm_d = 0.04, 0.04   # arm cross-section
    tiller_z0 = -0.22            # front face of tiller (within body)
    arm_y0    = 0.15             # tiller starts at top of body
    arm_h     = 0.75             # arm height → top at Y = 0.90
    arm_x_off = 0.13             # inner edge offset from centreline

    arm_l = o3d.geometry.TriangleMesh.create_box(arm_w, arm_h, arm_d)
    arm_l.translate([-arm_x_off - arm_w, arm_y0, tiller_z0])

    arm_r = o3d.geometry.TriangleMesh.create_box(arm_w, arm_h, arm_d)
    arm_r.translate([ arm_x_off,         arm_y0, tiller_z0])

    # Crossbar spans between the outer edges of both arms
    bar_w = 2 * (arm_x_off + arm_w)
    bar   = o3d.geometry.TriangleMesh.create_box(bar_w, arm_w, arm_d)
    bar.translate([-bar_w / 2, arm_y0 + arm_h - arm_w, tiller_z0])

    return body + fork_l + fork_r + arm_l + arm_r + bar


def load_carretilla(stl_path: str) -> o3d.geometry.TriangleMesh:
    """Load carretilla elevadora STL (mm) and place it in world frame.

    Normalisation applied:
      • scale 0.001  (mm → m)
      • centre X so vehicle is symmetric around X=0
      • keep Y as-is (STL already has Y_min=0, sits on floor)
      • translate Z so fork root lands at world Z=0
        (forks are at Z_max of the STL → fork root = Z_max − FORKLIFT_FORK_L)

    After transform:
      forks : Z ∈ [0,  FORKLIFT_FORK_L]  (+Z, toward cargo/pallet)
      body  : Z ∈ [−FORKLIFT_BODY_D, 0]  (−Z, away from cargo)
      X     : [−FORKLIFT_X_HALF, +FORKLIFT_X_HALF]
      Y     : [0, ~2.785]                 (on floor)

    If FORKLIFT_FORK_L or FORKLIFT_FORK_H are imprecise (physical estimates),
    verify visually with CloudCompare and adjust constants at top of file.
    """
    fk = o3d.io.read_triangle_mesh(stl_path)
    fk.scale(0.001, center=(0.0, 0.0, 0.0))   # mm → m
    fk.compute_vertex_normals()
    verts = np.asarray(fk.vertices)
    x_center = (verts[:, 0].max() + verts[:, 0].min()) / 2.0
    y_min    =  verts[:, 1].min()
    z_max    =  verts[:, 2].max()
    # Fork root assumed at z_max − FORKLIFT_FORK_L; shift it to world Z=0.
    z_translate = -(z_max - FORKLIFT_FORK_L)
    fk.translate([-x_center, -y_min, z_translate])
    return fk
