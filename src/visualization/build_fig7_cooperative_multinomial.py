"""Figure 7 — Sweep ρ pour Cooperative Multinomial (NB15, FISTA R custom).

Contraste avec la Fig 2 panneau B (NB11, multiview/glmnet IRLS qui ne
converge pas dès ρ > 0). NB15 converge à tout ρ et trouve `best_rho = 0.10`,
démontrant que le verrou ρ > 0 était purement numérique.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Couleurs cohérentes avec build_figures.py
COL_COOP_R   = "#16A085"   # cooperative multinomial R FISTA — vert sarcelle
COL_GREY     = "#7F8C8D"
COL_HIGHLIGHT = "#8E44AD"  # violet pour optimum

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Résultats NB15 (FISTA R custom, λ = 0.0847 fixe au best λ trouvé à ρ=0)
nb15 = {
    "rho":     [0.0,    0.1,    0.5,    1.0,    2.0,    5.0],
    "mean_ba": [0.7947, 0.8097, 0.7434, 0.6937, 0.6581, 0.6190],
    "sd":      [0.1365, 0.1601, 0.1607, 0.1340, 0.1709, 0.1888],
}
best_rho_idx = int(np.argmax(nb15["mean_ba"]))
best_rho     = nb15["rho"][best_rho_idx]
best_ba      = nb15["mean_ba"][best_rho_idx]
best_sd      = nb15["sd"][best_rho_idx]

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(nb15["rho"], nb15["mean_ba"], yerr=nb15["sd"],
            fmt="s-", capsize=6, color=COL_COOP_R, linewidth=2,
            markersize=11, markeredgecolor="black", markeredgewidth=0.5,
            label="Cooperative multinomial (NB15, FISTA R custom)")

# Annotation de chaque point
for rho, mu in zip(nb15["rho"], nb15["mean_ba"]):
    ax.annotate(f"{mu:.3f}", (rho, mu), xytext=(8, 10),
                textcoords="offset points", fontsize=9,
                color=COL_COOP_R, fontweight="bold")

# Mettre en évidence l'optimum
ax.axvline(x=best_rho, color=COL_HIGHLIGHT, linestyle=":",
           alpha=0.7, linewidth=1.5)
ax.annotate(rf"optimum $\rho^* = {best_rho}$"
            "\n"
            rf"CV = {best_ba:.3f} $\pm$ {best_sd:.3f}",
            (best_rho, best_ba),
            xytext=(60, -20), textcoords="offset points",
            fontsize=9.5, color=COL_HIGHLIGHT, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COL_HIGHLIGHT, lw=0.8))

ax.set_xlabel(r"$\rho$  (agreement penalty)")
ax.set_ylabel("Balanced accuracy CV (21 plis)")
ax.set_title("Cooperative Learning multinomial — sweep $\\rho$ avec solveur FISTA propre",
             pad=12)
ax.set_xticks([0, 0.1, 0.5, 1.0, 2.0, 5.0])
ax.set_xlim(-0.3, 5.5)
ax.set_ylim(0.4, 1.0)
ax.legend(loc="upper right")
ax.grid(alpha=0.25, linestyle="--")

# Encadré comparatif en bas à gauche
textstr = ("NB11 (multiview / IRLS glmnet)\n"
           r"$\rho = 0$ : 0.833  |  $\rho > 0$ : non-convergence"
           "\n\n"
           "NB15 (FISTA R proximal)\n"
           r"$\rho = 0$ : 0.795  |  $\rho^* = 0.10$ : 0.810"
           "\n\n"
           "Le gain $\\rho > 0$ existe ; le verrou\n"
           "de multiview était purement numérique.")
ax.text(0.98, 0.05, textstr, transform=ax.transAxes,
        fontsize=8.5, verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=COL_GREY, alpha=0.9))

plt.tight_layout()
plt.savefig(OUT / "fig7_cooperative_multinomial_rho.png",
            dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig7_cooperative_multinomial_rho.pdf",
            bbox_inches="tight")
plt.close()
print(f"✓ Figure 7 : {OUT / 'fig7_cooperative_multinomial_rho.png'}")
