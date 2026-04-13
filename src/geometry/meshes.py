"""
geometry/meshes.py  —  Mesh construction helpers.

Moved from generate_dataset.py (B9 refactor).
"""

from pathlib import Path

import numpy as np
import open3d as o3d

# EUR pallet dimensions (m)
EUR_W, EUR_H, EUR_D = 1.20, 0.144, 0.80


def make_pallet_mesh() -> o3d.geometry.TriangleMesh:
    """Standard EUR pallet: 1200×144×800 mm, centred on XZ at Y=0."""
    m = o3d.geometry.TriangleMesh.create_box(EUR_W, EUR_H, EUR_D)
    m.translate([-EUR_W / 2, 0.0, -EUR_D / 2])
    return m


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
    """
    Simplified traspaleta (pallet jack).
    Origin = body front face, floor level (Y=0, Z=0).
    Forks extend in +Z (toward cargo/pallet).
    Body extends in -Z (operator side).

    Local extents:
      Body:   X:-0.35..+0.35, Y:0..0.90, Z:-0.40..0
      Fork L: X:-0.30..-0.15, Y:0..0.08, Z:0..+1.15
      Fork R: X:+0.15..+0.30, Y:0..0.08, Z:0..+1.15
    """
    body = o3d.geometry.TriangleMesh.create_box(0.70, 0.90, 0.40)
    body.translate([-0.35, 0.0, -0.40])
    fork_l = o3d.geometry.TriangleMesh.create_box(0.15, 0.08, 1.15)
    fork_l.translate([-0.35, 0.0, 0.0])
    fork_r = o3d.geometry.TriangleMesh.create_box(0.15, 0.08, 1.15)
    fork_r.translate([ 0.20, 0.0, 0.0])
    return body + fork_l + fork_r


def load_forklift(stl_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load forklift STL (in mm) and convert to metres.
    STL Z=0 is the cab rear, Z=1.962 are the forks.
    We translate so forks sit just behind the pallet back edge (world Z≈-0.85m)
    and the cab is further back (world Z≈-2.8m).  No overlap with cargo.
    """
    fk = o3d.io.read_triangle_mesh(stl_path)
    fk.scale(0.001, center=(0.0, 0.0, 0.0))    # mm → m
    fk.compute_vertex_normals()
    verts = np.asarray(fk.vertices)
    y_min = verts[:, 1].min()
    # Sit on floor (Y) and push back so forks (Z=1.962) end up at world Z≈-0.85
    # → Z_translate = -0.85 - 1.962 = -2.812 ≈ -2.8
    fk.translate([0.0, -y_min, -2.8])
    return fk
