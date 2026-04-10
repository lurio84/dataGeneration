"""
app.py  —  Streamlit UI para el generador de dataset sintético.

Uso (desde src/):
    streamlit run app.py
"""

import sys
import json
import time
import base64
from pathlib import Path

import streamlit as st

# ── Importar módulos del proyecto ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import run_generation, CFG          # noqa: E402
from preview_grid import render_scene                      # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────

def _image_hover(container, png_path: Path, caption: str) -> None:
    """Imagen con hover de borde azul (CSS) y click para abrir en nueva pestaña."""
    data = base64.b64encode(png_path.read_bytes()).decode()
    src  = f"data:image/png;base64,{data}"
    html = f"""
<style>
.prev-img {{ width:100%; border-radius:4px; cursor:zoom-in; display:block;
    outline:2px solid transparent;
    transition: outline-color .15s ease, box-shadow .15s ease; }}
.prev-img:hover {{ outline-color:#4C9BE8; box-shadow:0 0 10px 3px rgba(76,155,232,.45); }}
.prev-cap {{ text-align:center; font-size:.82em; margin:3px 0 6px; opacity:.7; }}
</style>
<a href="{src}" target="_blank" style="display:block;">
  <img class="prev-img" src="{src}" />
</a>
<p class="prev-cap">{caption}</p>
"""
    container.markdown(html, unsafe_allow_html=True)


# ── Constantes ─────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).parent
PREVIEW_DIR = SRC_DIR.parent / "output" / "previews"

# Valores por defecto mapeados a las keys de session_state.
# Cada widget usa key=<nombre> → reseteable con un solo bucle.
DEFAULTS = {
    "n_samples":        CFG["n_samples"],
    "seed":             CFG["seed"],
    "output_dir":       CFG["output_dir"],
    "enable_floor":     CFG.get("enable_floor", True),
    "floor_extent_x":   float(CFG["floor_extent_x"]),
    "floor_extent_z":   float(CFG["floor_extent_z"]),
    "p_pallet":         float(CFG["p_pallet"]),
    "p_multi_cargo":    float(CFG["p_multi_cargo"]),
    "p_flat_cargo":     float(CFG["p_flat_cargo"]),
    "flat_min_h":       float(CFG["flat_min_h"]),
    "flat_max_h":       float(CFG["flat_max_h"]),
    "p_person":         float(CFG["p_person"]),
    "p_cylinder":       float(CFG["p_cylinder"]),
    "cyl_r":            (float(CFG["cyl_min_r"]), float(CFG["cyl_max_r"])),
    "cyl_h":            (float(CFG["cyl_min_h"]), float(CFG["cyl_max_h"])),
    "box_w":            (float(CFG["box_min_w"]), float(CFG["box_max_w"])),
    "box_d":            (float(CFG["box_min_d"]), float(CFG["box_max_d"])),
    "box_h":            (float(CFG["box_min_h"]), float(CFG["box_max_h"])),
    "noise_std":        float(CFG["noise_std"]),
    "dropout_ratio":    float(CFG["dropout_ratio"]),
    "outlier_ratio":    float(CFG["outlier_ratio"]),
    "voxel_size":       float(CFG["voxel_size"]),
    "local_outlier_std": float(CFG["local_outlier_std"]),
}

# ── Página ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Datageneration",
    page_icon="📦",
    layout="wide",
)
st.title("📦 Generador de dataset sintético")
st.caption("Nubes de puntos etiquetadas calibradas a FUSION3D · CATEC / INESARC")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — parámetros
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Configuración")

    if st.button("🔄 Restaurar valores por defecto", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    st.divider()

    # ── Dataset ────────────────────────────────────────────────────────────────
    with st.expander("📁 Dataset", expanded=True):
        n_samples = st.number_input(
            "Nº de escenas", min_value=1, max_value=2000, step=10, key="n_samples",
            value=DEFAULTS["n_samples"],
        )
        seed = st.number_input(
            "Seed", min_value=0, max_value=99999, step=1, key="seed",
            value=DEFAULTS["seed"],
        )
        output_dir = st.text_input(
            "Directorio de salida", key="output_dir",
            value=DEFAULTS["output_dir"],
        )

    # ── Composición de escena ──────────────────────────────────────────────────
    with st.expander("🧱 Composición de escena", expanded=True):
        enable_floor = st.checkbox(
            "Suelo activo", key="enable_floor",
            value=DEFAULTS["enable_floor"],
        )
        floor_extent_x = st.slider(
            "Suelo half-extent X (m)", 1.0, 6.0, step=0.5, key="floor_extent_x",
            value=DEFAULTS["floor_extent_x"], disabled=not enable_floor,
        )
        floor_extent_z = st.slider(
            "Suelo half-extent Z (m)", 1.0, 6.0, step=0.5, key="floor_extent_z",
            value=DEFAULTS["floor_extent_z"], disabled=not enable_floor,
        )
        p_pallet = st.slider(
            "Prob. pallet EUR", 0.0, 1.0, step=0.05, key="p_pallet",
            value=DEFAULTS["p_pallet"],
        )
        st.caption("⚠️ Experimental")
        p_multi_cargo = st.slider(
            "Prob. cargo múltiple (stacked/tandem)", 0.0, 1.0, step=0.05, key="p_multi_cargo",
            value=DEFAULTS["p_multi_cargo"],
        )
        p_flat_cargo = st.slider(
            "Prob. cargo plano/bajo (≤flat_max_h)", 0.0, 1.0, step=0.05, key="p_flat_cargo",
            value=DEFAULTS["p_flat_cargo"],
        )
        flat_min_h = st.number_input(
            "Altura mín. cargo plano (m)", min_value=0.01, max_value=0.15,
            step=0.01, key="flat_min_h", value=DEFAULTS["flat_min_h"],
        )
        flat_max_h = st.number_input(
            "Altura máx. cargo plano (m)", min_value=0.03, max_value=0.30,
            step=0.01, key="flat_max_h", value=DEFAULTS["flat_max_h"],
        )
        p_person = st.slider(
            "Prob. persona", 0.0, 1.0, step=0.05, key="p_person",
            value=DEFAULTS["p_person"],
        )
        p_cylinder = st.slider(
            "Prob. cilindro (en vez de caja)", 0.0, 1.0, step=0.05, key="p_cylinder",
            value=DEFAULTS["p_cylinder"],
        )

    # ── Dimensiones de cilindro ────────────────────────────────────────────────
    with st.expander("🛢 Dimensiones de cilindro (m)", expanded=False):
        cyl_r = st.slider(
            "Radio (min, max)", 0.10, 0.80, step=0.05, key="cyl_r",
            value=DEFAULTS["cyl_r"],
        )
        cyl_h = st.slider(
            "Altura Y (min, max)", 0.10, 2.00, step=0.05, key="cyl_h",
            value=DEFAULTS["cyl_h"],
        )

    # ── Dimensiones de caja ────────────────────────────────────────────────────
    with st.expander("📐 Dimensiones de caja (m)", expanded=True):
        box_w = st.slider(
            "Anchura X (min, max)", 0.10, 2.00, step=0.05, key="box_w",
            value=DEFAULTS["box_w"],
        )
        box_d = st.slider(
            "Profundidad Z (min, max)", 0.10, 2.00, step=0.05, key="box_d",
            value=DEFAULTS["box_d"],
        )
        box_h = st.slider(
            "Altura Y (min, max)", 0.10, 2.50, step=0.05, key="box_h",
            value=DEFAULTS["box_h"],
        )

    # ── Ruido del sensor ───────────────────────────────────────────────────────
    with st.expander("📡 Ruido del sensor", expanded=False):
        noise_std = st.slider(
            "Ruido gaussiano σ (m)", 0.005, 0.100, step=0.001,
            format="%.3f", key="noise_std", value=DEFAULTS["noise_std"],
        )
        dropout_ratio = st.slider(
            "Dropout de puntos", 0.00, 0.60, step=0.01,
            format="%.2f", key="dropout_ratio", value=DEFAULTS["dropout_ratio"],
        )
        outlier_ratio = st.slider(
            "Ratio outliers locales", 0.00, 0.20, step=0.005,
            format="%.3f", key="outlier_ratio", value=DEFAULTS["outlier_ratio"],
        )
        voxel_size = st.slider(
            "Voxel grid (m)", 0.005, 0.050, step=0.001,
            format="%.3f", key="voxel_size", value=DEFAULTS["voxel_size"],
        )
        local_outlier_std = st.slider(
            "σ dispersión outliers (m)", 0.010, 0.150, step=0.005,
            format="%.3f", key="local_outlier_std", value=DEFAULTS["local_outlier_std"],
        )

# ══════════════════════════════════════════════════════════════════════════════
# Construir cfg desde los widgets
# ══════════════════════════════════════════════════════════════════════════════

cfg = CFG.copy()
cfg.update({
    "n_samples":         int(n_samples),
    "seed":              int(seed),
    "output_dir":        output_dir,
    "enable_floor":      enable_floor,
    "floor_extent_x":    floor_extent_x,
    "floor_extent_z":    floor_extent_z,
    "p_pallet":          p_pallet,
    "p_multi_cargo":     p_multi_cargo,
    "p_flat_cargo":      p_flat_cargo,
    "flat_min_h":        flat_min_h,
    "flat_max_h":        flat_max_h,
    "p_person":          p_person,
    "p_forklift":        0.0,
    "p_cylinder":        p_cylinder,
    "cyl_min_r":         cyl_r[0],
    "cyl_max_r":         cyl_r[1],
    "cyl_min_h":         cyl_h[0],
    "cyl_max_h":         cyl_h[1],
    "box_min_w":         box_w[0],
    "box_max_w":         box_w[1],
    "box_min_d":         box_d[0],
    "box_max_d":         box_d[1],
    "box_min_h":         box_h[0],
    "box_max_h":         box_h[1],
    "noise_std":         noise_std,
    "dropout_ratio":     dropout_ratio,
    "outlier_ratio":     outlier_ratio,
    "voxel_size":        voxel_size,
    "local_outlier_std": local_outlier_std,
})

# ══════════════════════════════════════════════════════════════════════════════
# ÁREA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

col_gen, col_prev = st.columns([1, 1], gap="large")

# ── Panel izquierdo: generación ────────────────────────────────────────────────
with col_gen:
    st.subheader("▶ Generar dataset", anchor=False)

    st.caption(
        f"**{n_samples}** escenas · seed={seed} · "
        f"suelo={'sí' if enable_floor else 'no'} · "
        f"pallet={p_pallet:.0%} · "
        f"voxel={voxel_size*100:.1f} cm · "
        f"ruido={noise_std*100:.1f} cm"
    )

    if st.button("▶ Generar dataset", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Iniciando…")
        status_text  = st.empty()
        t0 = time.time()

        def update_progress(i, n):
            progress_bar.progress(i / n, text=f"Escena {i}/{n}")
            if i % max(1, n // 20) == 0 or i == n:
                status_text.caption(f"Escena {i}/{n} — {time.time()-t0:.1f}s")

        try:
            all_meta = run_generation(cfg, progress_cb=update_progress)
            elapsed  = time.time() - t0
            progress_bar.progress(1.0, text="Completado")
            pts_medio = sum(m["n_points"] for m in all_meta) // len(all_meta)
            st.success(
                f"✅ {len(all_meta)} escenas generadas en {elapsed:.1f}s  "
                f"(~{pts_medio:,} pts/escena)"
            )
            st.session_state["last_meta"] = all_meta
            st.session_state["last_out"]  = output_dir
        except Exception as e:
            st.error(f"Error: {e}")
            raise

    meta_path = Path(output_dir) / "metadata.json"
    if meta_path.exists():
        with st.expander("📄 Metadata (metadata.json)", expanded=False):
            with open(meta_path) as f:
                meta_data = json.load(f)
            st.json(meta_data[:10])
            if len(meta_data) > 10:
                st.caption(f"… {len(meta_data) - 10} escenas más en {meta_path}")

# ── Panel derecho: preview ─────────────────────────────────────────────────────
with col_prev:
    st.subheader("🖼 Vista previa", anchor=False)

    plys = sorted(Path(output_dir).glob("*.ply")) if Path(output_dir).exists() else []

    if not plys:
        st.info("Genera el dataset primero.")
    else:
        n_prev = st.slider(
            "Escenas a previsualizar", 1, min(20, len(plys)), min(5, len(plys)), step=1,
        )

        if st.button("🖼 Generar previews", use_container_width=True):
            PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            prog     = st.progress(0, text="Renderizando…")
            selected = plys[:n_prev]
            for idx, ply in enumerate(selected, 1):
                png_path = PREVIEW_DIR / f"{ply.stem}.png"
                render_scene(ply, png_path)
                prog.progress(idx / len(selected), text=f"Renderizando {idx}/{len(selected)}")
            prog.progress(1.0, text="Completado")
            st.session_state["preview_plys"] = selected

        preview_plys = st.session_state.get("preview_plys", [])
        if preview_plys:
            img_cols = st.columns(2)
            for idx, ply in enumerate(preview_plys):
                png_path = PREVIEW_DIR / f"{ply.stem}.png"
                if png_path.exists():
                    _image_hover(img_cols[idx % 2], png_path, ply.stem)
