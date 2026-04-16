#!/usr/bin/env python3
"""
Plots de evaluación del pipeline geométrico vs referencia Paula.
Leer comparison.csv y generar 4 PNGs en output/eval_real_20esc/plots/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "output/eval_real_20esc/comparison.csv"
OUT  = ROOT / "output/eval_real_20esc/plots"
OUT.mkdir(parents=True, exist_ok=True)

# ── Cargar datos ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df["ratio"]            = pd.to_numeric(df["ratio"],            errors="coerce")
df["vol_paula_m3"]     = pd.to_numeric(df["vol_paula_m3"],     errors="coerce")
df["vol_pipeline_m3"]  = pd.to_numeric(df["vol_pipeline_m3"],  errors="coerce")
df["esc_num"] = df["escenario"].str.extract(r"(\d+)$").astype(int)
df["cap_num"] = df["captura"].str.extract(r"Captura_(\d+)").astype(int)
df["label"]   = df.apply(lambda r: f"E{r.esc_num:02d}C{r.cap_num:02d}", axis=1)

valid = df.dropna(subset=["ratio", "vol_paula_m3", "vol_pipeline_m3"])
valid = valid[valid["vol_paula_m3"] > 0]

GLOBAL_MEDIAN = valid["ratio"].median()
N_ESC = 20
cmap  = plt.get_cmap("tab20")
esc_colors = {i: cmap(i / N_ESC) for i in range(1, N_ESC + 1)}

# ═══════════════════════════════════════════════════════════════════
# a) SCATTER log-log
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 9))

for esc_n, grp in valid.groupby("esc_num"):
    ax.scatter(grp["vol_paula_m3"], grp["vol_pipeline_m3"],
               color=esc_colors[esc_n], s=40, alpha=0.75, zorder=3,
               label=f"Esc {esc_n:02d}")

# Líneas de referencia
lim_lo = valid[["vol_paula_m3","vol_pipeline_m3"]].min().min() * 0.8
lim_hi = valid[["vol_paula_m3","vol_pipeline_m3"]].max().max() * 1.3
xs = np.array([lim_lo, lim_hi])
ax.plot(xs, xs,             color="black", ls="--", lw=1.5, label="Perfecto (1:1)", zorder=4)
ax.plot(xs, GLOBAL_MEDIAN*xs, color="gray",  ls="--", lw=1.5,
        label=f"Mediana actual ({GLOBAL_MEDIAN:.2f}×)", zorder=4)

# Anotar outliers (ratio > 3 o < 0.3)
for _, row in valid.iterrows():
    if row["ratio"] > 3 or row["ratio"] < 0.3:
        ax.annotate(row["label"],
                    xy=(row["vol_paula_m3"], row["vol_pipeline_m3"]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=6.5, color="crimson",
                    arrowprops=dict(arrowstyle="-", color="crimson", lw=0.5))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
ax.set_xlabel("Volumen Paula [m³]", fontsize=12)
ax.set_ylabel("Volumen pipeline [m³]", fontsize=12)
ax.set_title("Volumen pipeline vs Paula (escala log-log)", fontsize=14, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc="upper left", fontsize=7, ncol=2,
          title="Escenario", title_fontsize=8, framealpha=0.9)
plt.tight_layout()
fig.savefig(OUT / "scatter_vol.png", dpi=150)
plt.close(fig)
print(f"[OK] scatter_vol.png  ({len(valid)} puntos, {(valid['ratio']>3).sum()} outliers >3, {(valid['ratio']<0.3).sum()} <0.3)")

# ═══════════════════════════════════════════════════════════════════
# b) BOXPLOT ratio por escenario
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(17, 7))

groups = []
positions = []
colors_box = []
for esc_n in range(1, N_ESC + 1):
    sub = valid[valid["esc_num"] == esc_n]["ratio"].dropna()
    if len(sub) == 0:
        continue
    groups.append(sub.values)
    positions.append(esc_n)
    med = sub.median()
    colors_box.append("#4caf50" if 0.5 <= med <= 1.2 else "#ff9800")

bp = ax.boxplot(groups, positions=positions, widths=0.6,
                patch_artist=True, showfliers=True,
                flierprops=dict(marker=".", markersize=5, linestyle="none", alpha=0.7),
                medianprops=dict(color="black", lw=2))
for patch, c in zip(bp["boxes"], colors_box):
    patch.set_facecolor(c); patch.set_alpha(0.75)

ax.axhline(1.0, color="black", lw=1.5, ls="-",  label="Perfecto (1.0)")
ax.axhline(GLOBAL_MEDIAN, color="dimgray", lw=1.5, ls="--",
           label=f"Mediana global ({GLOBAL_MEDIAN:.2f})")
ax.axhline(0.4, color="salmon", lw=1, ls=":",  label="Umbral inferior (0.4)")
ax.axhline(1.5, color="salmon", lw=1, ls=":",  label="Umbral superior (1.5)")

ax.set_yscale("log")
ax.set_xticks(range(1, N_ESC + 1))
ax.set_xticklabels([f"{i:02d}" for i in range(1, N_ESC + 1)], fontsize=9)
ax.set_xlabel("Escenario", fontsize=12)
ax.set_ylabel("Ratio  pipeline / Paula  (log)", fontsize=12)
ax.set_title("Distribución del ratio pipeline/Paula por escenario", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.grid(True, axis="y", ls=":", alpha=0.4)

legend_extra = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor="#4caf50",
           markersize=10, label="Mediana 0.5–1.2 (bien calibrado)"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="#ff9800",
           markersize=10, label="Mediana fuera de rango"),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + legend_extra,
          labels=labels + [p.get_label() for p in legend_extra],
          loc="upper right", fontsize=8.5)
plt.tight_layout()
fig.savefig(OUT / "boxplot_ratio_escenario.png", dpi=150)
plt.close(fig)
print("[OK] boxplot_ratio_escenario.png")

# ═══════════════════════════════════════════════════════════════════
# c) HISTOGRAMA de ratios (excluye > 5)
# ═══════════════════════════════════════════════════════════════════
RATIO_CAP = 5.0
filt  = valid[valid["ratio"] <= RATIO_CAP]["ratio"]
n_excl = len(valid) - len(filt)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(filt, bins=30, color="steelblue", edgecolor="white", alpha=0.85)

med_f  = filt.median()
mean_f = filt.mean()
ax.axvline(med_f,  color="red",    lw=2,   ls="-",  label=f"Mediana = {med_f:.2f}")
ax.axvline(mean_f, color="orange", lw=2,   ls="--", label=f"Media   = {mean_f:.2f}")
ax.axvline(1.0,    color="black",  lw=1.5, ls=":",  label="Perfecto (1.0)")

txt = (f"Incluidas: {len(filt)}\n"
       f"Excluidas (ratio > {RATIO_CAP:.0f}): {n_excl}")
ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
        fontsize=9.5, bbox=dict(boxstyle="round", fc="white", alpha=0.85))

ax.set_xlabel("Ratio  (pipeline / Paula)", fontsize=12)
ax.set_ylabel("Número de capturas", fontsize=12)
ax.set_title(f"Distribución de ratios (excluye ratio > {RATIO_CAP:.0f})", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, axis="y", ls=":", alpha=0.4)
plt.tight_layout()
fig.savefig(OUT / "histograma_ratio.png", dpi=150)
plt.close(fig)
print(f"[OK] histograma_ratio.png  (N={len(filt)}, excluidas={n_excl})")

# ═══════════════════════════════════════════════════════════════════
# d) BARPLOT volumen medio por escenario (excluye ratio > 5)
# ═══════════════════════════════════════════════════════════════════
fair  = valid[valid["ratio"] <= RATIO_CAP]
stats = fair.groupby("esc_num").agg(
    mean_paula=("vol_paula_m3",   "mean"),
    std_paula=("vol_paula_m3",    "std"),
    mean_pipe=("vol_pipeline_m3", "mean"),
    std_pipe=("vol_pipeline_m3",  "std"),
    n=("ratio", "count"),
).reset_index()

x = np.arange(len(stats))
w = 0.38
fig, ax = plt.subplots(figsize=(17, 7))
ax.bar(x - w/2, stats["mean_paula"], w,
       yerr=stats["std_paula"], capsize=3,
       color="steelblue", alpha=0.85, label="Paula (referencia)")
ax.bar(x + w/2, stats["mean_pipe"],  w,
       yerr=stats["std_pipe"],  capsize=3,
       color="coral",     alpha=0.85, label="Pipeline (nuestro)")

ax.set_xticks(x)
ax.set_xticklabels([f"{int(r.esc_num):02d}" for _, r in stats.iterrows()], fontsize=9)
ax.set_xlabel("Escenario", fontsize=12)
ax.set_ylabel("Volumen medio [m³]", fontsize=12)
ax.set_title("Volumen medio por escenario: pipeline vs Paula\n(excluye capturas con ratio > 5)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, axis="y", ls=":", alpha=0.4)

# Anotar ratio medio encima de cada par de barras
for xi, (_, row) in zip(x, stats.iterrows()):
    r_mean = row["mean_pipe"] / row["mean_paula"] if row["mean_paula"] > 0 else float("nan")
    if not np.isnan(r_mean):
        y_top = max(row["mean_paula"] + (row["std_paula"] or 0),
                    row["mean_pipe"]  + (row["std_pipe"]  or 0))
        ax.text(xi, y_top * 1.02, f"{r_mean:.2f}×",
                ha="center", va="bottom", fontsize=7.5, color="dimgray")

plt.tight_layout()
fig.savefig(OUT / "barplot_vol_escenario.png", dpi=150)
plt.close(fig)
print(f"[OK] barplot_vol_escenario.png  ({len(stats)} escenarios)")

print(f"\nTodos los plots en: {OUT}")
