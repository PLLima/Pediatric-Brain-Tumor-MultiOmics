"""Figures de synthèse v2 — intègre tous les nouveaux résultats Tenenhaus.

Génère :
- fig8_class_weights.png   : NB17 — comparaison schémas de poids
- fig9_stability_bootstrap.png : NB18 — pi_hat des variables stables
- fig10_cgh_spatial.png    : NB19/NB20 — CGH groupé vs Fused Lasso
- fig11_final_comparison.png : tableau visuel final avec toutes les méthodes
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import json

OUT = Path(__file__).resolve().parent / "figures"
SYN = Path(__file__).resolve().parent

# Couleurs
COL_SGCCA   = "#27AE60"
COL_COOP    = "#3498DB"
COL_MULTI_G = "#9B59B6"
COL_MULTI_U = "#E67E22"
COL_COOP_R  = "#16A085"
COL_TV      = "#E91E63"   # NB20 Fused TV
COL_GLASSO  = "#FF9800"   # NB19 Group Lasso
COL_WEIGHT  = "#7E57C2"   # NB17 weights
COL_GREY    = "#7F8C8D"
COL_BAD     = "#C0392B"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
})

# =====================================================================
# Charger les RDS en passant par Rscript
# =====================================================================
def load_rds(path, *fields):
    """Lit un RDS et renvoie un dict des champs nommés."""
    script = f"""
x <- readRDS('{path}')
cat(jsonlite::toJSON(x, auto_unbox=TRUE, force=TRUE, na='null', null='null'))
"""
    r = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    return json.loads(r.stdout)


# =====================================================================
# Figure 8 — Class weights (NB17)
# =====================================================================
nb17 = load_rds(SYN / "nb17_class_weights_results.rds")
cv_df = nb17["cv"]
test_df = nb17["test"]

schemes = ["none", "inv_prevalence", "sqrt_inv"]
sch_labels = ["sans poids", "inv. prévalence", "racine inverse"]
families = ["multinomial", "ovr"]
fam_labels = {"multinomial": "Multinomial ungrouped", "ovr": "OvR binomial"}
fam_colors = {"multinomial": COL_MULTI_U, "ovr": COL_COOP}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (A) CV bal_acc
ax = axes[0]
x = np.arange(len(schemes)); w = 0.36
for j, fam in enumerate(families):
    means = [next(r["mean_bal_acc"] for r in cv_df
                  if r["scheme"]==s and r["family"]==fam) for s in schemes]
    sds   = [next(r["sd_bal_acc"]   for r in cv_df
                  if r["scheme"]==s and r["family"]==fam) for s in schemes]
    pos = x - w/2 + j*w
    ax.bar(pos, means, yerr=sds, width=w, capsize=4,
           color=fam_colors[fam], edgecolor="black", linewidth=0.6,
           label=fam_labels[fam], alpha=0.88)
    for xi, mu in zip(pos, means):
        ax.text(xi, mu + 0.02, f"{mu:.3f}", ha="center",
                fontsize=8.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(sch_labels)
ax.set_ylabel("Balanced accuracy CV (21 plis)")
ax.set_title("(A)  CV — gain par schéma de poids", pad=8, fontweight="bold")
ax.set_ylim(0.5, 1.0); ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.25, linestyle="--")

# (B) Test bal_acc + midl recall
ax = axes[1]
test_bas = {(r["scheme"], r["family"]): r["test_bal_acc"] for r in test_df}
midls    = {(r["scheme"], r["family"]): int(r["midl_correct"]) for r in test_df}
for j, fam in enumerate(families):
    vals = [test_bas[(s, fam)] for s in schemes]
    midl_vals = [midls[(s, fam)] for s in schemes]
    pos = x - w/2 + j*w
    bars = ax.bar(pos, vals, width=w, color=fam_colors[fam],
                  edgecolor="black", linewidth=0.6,
                  label=fam_labels[fam], alpha=0.88)
    for xi, v, m in zip(pos, vals, midl_vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center",
                fontsize=8.5, fontweight="bold")
        ax.text(xi, 0.10, f"midl\n{m}/3", ha="center",
                fontsize=8, color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(sch_labels)
ax.set_ylabel("Balanced accuracy test (n=14)")
ax.set_title("(B)  Test — récupération midl par les poids", pad=8, fontweight="bold")
ax.set_ylim(0, 1.05); ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.25, linestyle="--")

fig.suptitle("Gestion du déséquilibre des classes par pondération de la log-vraisemblance",
             fontsize=12.5)
plt.tight_layout()
plt.savefig(OUT / "fig8_class_weights.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig8_class_weights.pdf", bbox_inches="tight")
plt.close()
print(f"✓ Figure 8 : {OUT / 'fig8_class_weights.png'}")


# =====================================================================
# Figure 9 — Stability bootstrap (NB18)
# =====================================================================
nb18 = load_rds(SYN / "nb18_stability_results.rds")

# pi_max_multi et pi_max_ovr sont des listes de fréquences par variable
pi_multi = nb18["pi_max_multi"]
pi_ovr   = nb18["pi_max_ovr"]

# Convertir en arrays + tri
if isinstance(pi_multi, dict):
    names_multi = list(pi_multi.keys()); vals_multi = list(pi_multi.values())
    names_ovr   = list(pi_ovr.keys());   vals_ovr   = list(pi_ovr.values())
else:
    names_multi = nb18["pi_max_multi"]; names_ovr = nb18["pi_max_ovr"]

# Workaround : récupérer noms via R direct
r_script = """
x <- readRDS('%s/nb18_stability_results.rds')
df_multi <- data.frame(name=names(x$pi_max_multi), val=as.numeric(x$pi_max_multi))
df_ovr   <- data.frame(name=names(x$pi_max_ovr),   val=as.numeric(x$pi_max_ovr))
df_multi <- df_multi[order(-df_multi$val), ][1:30, ]
df_ovr   <- df_ovr[order(-df_ovr$val), ][1:30, ]
out <- list(
  multi_name = as.character(df_multi$name),
  multi_val  = as.numeric(df_multi$val),
  ovr_name   = as.character(df_ovr$name),
  ovr_val    = as.numeric(df_ovr$val))
cat(jsonlite::toJSON(out, auto_unbox=FALSE))
""" % SYN
r = subprocess.run(["Rscript", "-e", r_script], capture_output=True, text=True)
top = json.loads(r.stdout)
names_m = top["multi_name"]; vals_m = top["multi_val"]
names_o = top["ovr_name"];   vals_o = top["ovr_val"]

# Couleur GE vs CGH
def color_of(n): return COL_COOP if n.startswith("GE__") else COL_BAD

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, names, vals, title in [(axes[0], names_o, vals_o, "OvR binomial Lasso"),
                                (axes[1], names_m, vals_m, "Multinomial ungrouped Lasso")]:
    cols = [color_of(n) for n in names]
    short = [n.replace("GE__","").replace("CGH__","Chr.")[:18] for n in names]
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.3, alpha=0.85)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, fontweight="bold")
    ax.axvline(x=0.6, color=COL_GREY, linestyle=":", alpha=0.7)
    ax.axvline(x=0.8, color="black",  linestyle=":", alpha=0.7)
    ax.text(0.61, len(names)-0.5, " seuil 0.6", fontsize=7.5, color=COL_GREY)
    ax.text(0.81, len(names)-0.5, " seuil 0.8", fontsize=7.5, color="black")
    ax.set_yticks(y); ax.set_yticklabels(short, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\hat\pi_j$ (fréquence de sélection sur B=200 bootstraps)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=COL_COOP, label="GE"),
           mpatches.Patch(color=COL_BAD,  label="CGH")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.04),
           ncol=2, frameon=False, fontsize=10)
fig.suptitle("Stabilité de la sélection de variables — Meinshausen-Bühlmann 2010",
             fontsize=12.5, y=1.0)
plt.tight_layout()
plt.subplots_adjust(bottom=0.10)
plt.savefig(OUT / "fig9_stability_bootstrap.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig9_stability_bootstrap.pdf", bbox_inches="tight")
plt.close()
print(f"✓ Figure 9 : {OUT / 'fig9_stability_bootstrap.png'}")


# =====================================================================
# Figure 10 — CGH spatial : profil moyen + Fused Lasso beta
# =====================================================================
nb20 = load_rds(SYN / "nb20_fused_lasso_results.rds")
# Récupérer beta_cgh_per_class via R direct
r_script = """
x <- readRDS('%s/nb20_fused_lasso_results.rds')
classes <- names(x$beta_cgh_per_class)
out <- list()
for (cl in classes) {
  out[[cl]] <- list(
    beta = unname(x$beta_cgh_per_class[[cl]]$beta),
    gamma = unname(x$beta_cgh_per_class[[cl]]$gamma),
    n_breakpoints = x$beta_cgh_per_class[[cl]]$n_breakpoints,
    n_active_segs = x$beta_cgh_per_class[[cl]]$n_active_segs,
    n_active_ge = x$beta_cgh_per_class[[cl]]$n_active_ge)
}
out$cgh_order <- x$cgh_order
cat(jsonlite::toJSON(out, auto_unbox=TRUE, force=TRUE))
""" % SYN
r = subprocess.run(["Rscript", "-e", r_script], capture_output=True, text=True)
data20 = json.loads(r.stdout)

cgh_order = data20.pop("cgh_order")
classes = list(data20.keys())
class_cols = {"cort": COL_COOP, "dipg": "#F39C12", "midl": COL_BAD}

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
for ax, cl in zip(axes, classes):
    beta = np.array(data20[cl]["beta"])
    ax.axhline(y=0, color="black", linewidth=0.3, alpha=0.4)
    ax.plot(cgh_order, beta, color=class_cols[cl], linewidth=1.0)
    ax.fill_between(cgh_order, 0, beta, color=class_cols[cl], alpha=0.3,
                     step="mid")
    # marquer les sauts (gamma != 0)
    gamma = np.array(data20[cl]["gamma"])
    breaks = np.where(np.abs(gamma[1:]) > 1e-8)[0] + 1
    if len(breaks):
        ax.scatter(np.array(cgh_order)[breaks], beta[breaks],
                   s=10, color="black", zorder=5)
    n_b = data20[cl]["n_breakpoints"]
    n_a = data20[cl]["n_active_segs"]
    ax.set_title(f"Classe {cl}  —  {n_b} sauts TV  |  {n_a} segments avec coef. ≠ 0",
                 fontweight="bold", color=class_cols[cl], pad=4)
    ax.set_ylabel(r"$\hat\beta_{\mathrm{CGH}}$")
    ax.grid(alpha=0.2, linestyle="--")
axes[-1].set_xlabel("ID segment CGH (ordre numérique, proxy adjacence génomique)")

fig.suptitle("NB20 — Fused Lasso : coefficients CGH constants par morceaux le long du génome",
             fontsize=12.5, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig10_cgh_spatial.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig10_cgh_spatial.pdf", bbox_inches="tight")
plt.close()
print(f"✓ Figure 10 : {OUT / 'fig10_cgh_spatial.png'}")


# =====================================================================
# Figure 11 — Tableau final visuel : toutes les méthodes
# =====================================================================
all_methods = [
    # (label, color, CV mean, CV sd, test, midl_test)
    ("SGCCA + LDA",                      COL_SGCCA,   0.829, 0.133, 0.924, 2),
    ("Cooperative OvR (NB11)",           COL_COOP,    0.833, 0.129, 0.924, 2),
    ("Multinomial grouped (NB14a)",      COL_MULTI_G, 0.784, 0.096, 0.773, 0),
    ("Multinomial ungrouped (NB14c)",    COL_MULTI_U, 0.838, 0.123, 0.773, 0),
    ("Cooperative multi NB15 (ρ=0.1)",   COL_COOP_R,  0.810, 0.160, 0.771, 0),
    ("Multi + weights inv_prev (NB17)",  COL_WEIGHT,  0.830, 0.118, 0.924, 2),
    ("Group Lasso CGH (NB19, chr.)",     COL_GLASSO,  np.nan, 0,    np.nan, 0),
    ("Fused Lasso TV (NB20)",            COL_TV,      0.000, 0,    0.847, 1),
]
# Charger NB19/NB20
try:
    nb19 = load_rds(SYN / "nb19_group_lasso_results.rds")
    all_methods[6] = ("Group Lasso CGH (NB19)", COL_GLASSO,
                      np.nan, 0, nb19["test_bal_acc"][0] if isinstance(nb19["test_bal_acc"], list) else nb19["test_bal_acc"],
                      int(nb19["midl_test_correct"][0] if isinstance(nb19["midl_test_correct"], list) else nb19["midl_test_correct"]))
except Exception:
    pass
# NB20 CV chargée
all_methods[7] = ("Fused Lasso TV (NB20)", COL_TV,
                  nb20["cv"]["mean_bal_acc"], nb20["cv"]["sd_bal_acc"],
                  nb20["test_bal_acc"], int(nb20["midl_test_correct"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
labels = [m[0] for m in all_methods]
colors = [m[1] for m in all_methods]
cv_m   = [m[2] for m in all_methods]
cv_s   = [m[3] for m in all_methods]
test_m = [m[4] for m in all_methods]
midl_m = [m[5] for m in all_methods]

# CV
ax = axes[0]
y = np.arange(len(labels))
ax.barh(y, cv_m, xerr=cv_s, color=colors, edgecolor="black",
        linewidth=0.5, alpha=0.88, capsize=3)
for yi, mu in zip(y, cv_m):
    if not np.isnan(mu) and mu > 0:
        ax.text(mu + 0.04, yi, f"{mu:.3f}", va="center",
                fontsize=8.5, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("CV bal_acc (21 plis)")
ax.set_title("Cross-validation", pad=8, fontweight="bold")
ax.set_xlim(0.3, 1.05); ax.grid(axis="x", alpha=0.2, linestyle="--")

# Test + midl
ax = axes[1]
ax.barh(y, test_m, color=colors, edgecolor="black",
        linewidth=0.5, alpha=0.88)
for yi, v, m in zip(y, test_m, midl_m):
    if not np.isnan(v) and v > 0:
        ax.text(v + 0.04, yi, f"{v:.3f}  (midl {m}/3)", va="center",
                fontsize=8.5, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([])
ax.invert_yaxis()
ax.set_xlabel("Test bal_acc (n=14)")
ax.set_title("Test set tenu à part", pad=8, fontweight="bold")
ax.set_xlim(0.3, 1.15); ax.grid(axis="x", alpha=0.2, linestyle="--")

fig.suptitle("Comparaison finale — 8 pipelines évalués sur le même protocole",
             fontsize=12.5, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig11_final_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig11_final_comparison.pdf", bbox_inches="tight")
plt.close()
print(f"✓ Figure 11 : {OUT / 'fig11_final_comparison.png'}")

print("\nToutes les figures ont été régénérées.")
