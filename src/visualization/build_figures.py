"""Figures du document de synthèse — v3.1, post-NB14c, sans décorations.

4 pipelines comparés :
- NB09  SGCCA + LDA
- NB11  Cooperative Lasso OvR (ρ=0)
- NB14a LogReg multinomial Lasso GROUPED
- NB14c LogReg multinomial Lasso UNGROUPED
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

COL_COOP    = "#3498DB"
COL_MULTI_G = "#9B59B6"
COL_MULTI_U = "#E67E22"
COL_SGCCA   = "#27AE60"
COL_GREY    = "#7F8C8D"
COL_BAD     = "#C0392B"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ════════════════════════════════════════════════════════════════
# Données
# ════════════════════════════════════════════════════════════════

nb09 = {
    "Set":     list(range(1, 11)),
    "s_GE":    [0.200, 0.179, 0.157, 0.136, 0.115, 0.093, 0.072, 0.051, 0.029, 0.008],
    "s_CGH":   [0.200, 0.181, 0.162, 0.143, 0.124, 0.105, 0.086, 0.067, 0.048, 0.029],
    "mean_ba": [0.731, 0.726, 0.734, 0.734, 0.764, 0.785, 0.792, 0.829, 0.829, 0.718],
    "sd":      [0.161, 0.167, 0.174, 0.169, 0.163, 0.173, 0.150, 0.133, 0.143, 0.107],
}
nb09_cv_mean  = 0.829
nb09_cv_sd    = 0.133
nb09_test_ba  = 0.924
nb09_cm       = np.array([[5,0,0],[0,6,0],[0,1,2]])
nb09_vars_ge  = 68
nb09_vars_cgh = 11

nb11 = {
    "rho":       [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
    "mean_ba":   [0.833, np.nan, np.nan, np.nan, np.nan, np.nan],
    "sd":        [0.129, np.nan, np.nan, np.nan, np.nan, np.nan],
    "converged": [True, False, False, False, False, False],
}
nb11_cv_mean  = 0.833
nb11_cv_sd    = 0.129
nb11_test_ba  = 0.924
nb11_cm       = np.array([[5,0,0],[0,6,0],[0,1,2]])
nb11_vars_ge  = 42
nb11_vars_cgh = 0

nb14a = {
    "alpha":   [0.0,   0.5,   1.0],
    "mean_ba": [0.706, 0.781, 0.784],
    "sd":      [0.088, 0.120, 0.096],
}
nb14a_cv_mean   = 0.784
nb14a_cv_sd     = 0.096
nb14a_test_ba   = 0.773
nb14a_cm        = np.array([[5,0,1],[0,6,2],[0,0,0]])
nb14a_vars_ge   = 27
nb14a_vars_cgh  = 0

nb14c = {
    "alpha":       [0.0,   0.5,   1.0],
    "mean_ba":     [0.699, 0.794, 0.838],
    "sd":          [0.097, 0.120, 0.123],
    "midl_recall": [0.048, 0.238, 0.405],
}
nb14c_cv_mean   = 0.838
nb14c_cv_sd     = 0.123
nb14c_test_ba   = 0.773
nb14c_cm        = np.array([[5,0,1],[0,6,2],[0,0,0]])
nb14c_vars_ge_per_class = {"cort": 9, "dipg": 8, "midl": 10}
nb14c_intercepts        = {"cort": -1.931, "dipg": +1.292, "midl": +0.639}

# ════════════════════════════════════════════════════════════════
# Figure 1 — Performance comparée (CV + test)
# ════════════════════════════════════════════════════════════════

methods = ["SGCCA + LDA\n(NB09)",
           "Cooperative\nLasso OvR\n(NB11)",
           "Multinomial\nLasso GROUPED\n(NB14a)",
           "Multinomial\nLasso UNGROUPED\n(NB14c)"]
colors  = [COL_SGCCA, COL_COOP, COL_MULTI_G, COL_MULTI_U]
cv_means = [nb09_cv_mean, nb11_cv_mean, nb14a_cv_mean, nb14c_cv_mean]
cv_sds   = [nb09_cv_sd,   nb11_cv_sd,   nb14a_cv_sd,   nb14c_cv_sd]
test_bas = [nb09_test_ba, nb11_test_ba, nb14a_test_ba, nb14c_test_ba]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
x = np.arange(len(methods))
ax.bar(x, cv_means, yerr=cv_sds, capsize=7, color=colors, alpha=0.88,
       edgecolor="black", linewidth=0.7, width=0.62)
for i, (mu, sd) in enumerate(zip(cv_means, cv_sds)):
    ax.text(i, mu + sd + 0.018, f"{mu:.3f}\n± {sd:.3f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Balanced accuracy (CV 21 folds)")
ax.set_title("Cross-validation 7-fold × 3 (train, n=39)", pad=10)
ax.set_ylim(0.3, 1.05)
ax.grid(axis="y", alpha=0.25, linestyle="--")

ax = axes[1]
ax.bar(x, test_bas, color=colors, alpha=0.88,
       edgecolor="black", linewidth=0.7, width=0.62)
for i, v in enumerate(test_bas):
    ax.text(i, v + 0.018, f"{v:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Balanced accuracy (test, n=14)")
ax.set_title("Test set tenu à part", pad=10)
ax.set_ylim(0.3, 1.05)
ax.grid(axis="y", alpha=0.25, linestyle="--")

fig.suptitle("Performance des quatre approches sparses supervisées", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig1_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig1_comparison.pdf", bbox_inches="tight")
plt.close()
print(f"✓ Figure 1 : {OUT / 'fig1_comparison.png'}")

# ════════════════════════════════════════════════════════════════
# Figure 2 — Sensibilité aux hyperparamètres (3 panneaux)
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

# (A) SGCCA — sparsité
ax = axes[0]
ax.errorbar(nb09["Set"], nb09["mean_ba"], yerr=nb09["sd"], fmt="o-",
            capsize=5, color=COL_SGCCA, linewidth=2, markersize=8)
ax.set_xlabel("Set de sparsité")
ax.set_ylabel("Balanced accuracy CV")
ax.set_title("(A)  SGCCA + LDA (NB09)", pad=8, fontweight="bold")
ax.set_xticks(nb09["Set"])
ax.set_ylim(0.4, 1.0)
ax.grid(alpha=0.25, linestyle="--")

# (B) Cooperative — ρ (non-convergence pour ρ > 0)
ax = axes[1]
xs   = nb11["rho"]; mus = nb11["mean_ba"]; sds = nb11["sd"]; conv = nb11["converged"]
xs_c  = [x for x, c in zip(xs, conv) if c]
mus_c = [m for m, c in zip(mus, conv) if c]
sds_c = [s for s, c in zip(sds, conv) if c]
ax.errorbar(xs_c, mus_c, yerr=sds_c, fmt="o", capsize=6,
            color=COL_COOP, linewidth=2, markersize=11)
xs_nc = [x for x, c in zip(xs, conv) if not c]
for x in xs_nc:
    ax.scatter([x], [0.5], marker="x", s=140, color=COL_BAD, linewidth=2.5)
ax.text(0.5, 0.04, "✗ : non-convergence du solveur IRLS",
        transform=ax.transAxes, ha="center", fontsize=8.8,
        style="italic", color=COL_BAD)
ax.set_xlabel(r"$\rho$")
ax.set_ylabel("Balanced accuracy CV")
ax.set_title("(B)  Cooperative Lasso OvR (NB11)", pad=8, fontweight="bold")
ax.set_xticks([0, 0.1, 0.5, 1.0, 2.0, 5.0])
ax.set_xlim(-0.3, 5.5)
ax.set_ylim(0.4, 1.0)
ax.grid(alpha=0.25, linestyle="--")

# (C) Multinomial — α × type.multinomial
ax = axes[2]
xs = nb14a["alpha"]
ax.errorbar(xs, nb14a["mean_ba"], yerr=nb14a["sd"], fmt="o-",
            capsize=6, color=COL_MULTI_G, linewidth=2, markersize=10,
            label="grouped (NB14a)")
ax.errorbar(xs, nb14c["mean_ba"], yerr=nb14c["sd"], fmt="s-",
            capsize=6, color=COL_MULTI_U, linewidth=2, markersize=10,
            label="ungrouped (NB14c)")
for x, mu in zip(xs, nb14c["mean_ba"]):
    ax.annotate(f"{mu:.3f}", (x, mu), xytext=(8, 8), textcoords="offset points",
                fontsize=9, color=COL_MULTI_U, fontweight="bold")
for x, mu in zip(xs, nb14a["mean_ba"]):
    ax.annotate(f"{mu:.3f}", (x, mu), xytext=(8, -16), textcoords="offset points",
                fontsize=9, color=COL_MULTI_G)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Balanced accuracy CV")
ax.set_title("(C)  Multinomial Lasso (NB14)", pad=8, fontweight="bold")
ax.set_xticks([0, 0.5, 1.0])
ax.set_xlim(-0.1, 1.15)
ax.set_ylim(0.4, 1.0)
ax.legend(loc="lower right")
ax.grid(alpha=0.25, linestyle="--")

fig.suptitle("Sensibilité aux hyperparamètres, CV 7-fold × 3", fontsize=13)
plt.savefig(OUT / "fig2_hp_sensitivity.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig2_hp_sensitivity.pdf", bbox_inches="tight")
plt.close()
print(f"✓ Figure 2 : {OUT / 'fig2_hp_sensitivity.png'}")

# ════════════════════════════════════════════════════════════════
# Figure 3 — Matrices de confusion (4 méthodes)
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
labels = ["cort", "dipg", "midl"]
data = [
    (axes[0], "SGCCA + LDA\n(NB09)",             nb09_cm,  nb09_test_ba,  COL_SGCCA),
    (axes[1], "Cooperative OvR\n(NB11)",         nb11_cm,  nb11_test_ba,  COL_COOP),
    (axes[2], "Multinomial GROUPED\n(NB14a)",    nb14a_cm, nb14a_test_ba, COL_MULTI_G),
    (axes[3], "Multinomial UNGROUPED\n(NB14c)",  nb14c_cm, nb14c_test_ba, COL_MULTI_U),
]
for ax, title, cm, bal, col in data:
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=6, aspect="equal")
    for i in range(3):
        for j in range(3):
            v = cm[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > 3 else "black",
                    fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels([f"pred_{l}" for l in labels], fontsize=9)
    ax.set_yticklabels([f"true_{l}" for l in labels], fontsize=9)
    ax.set_title(f"{title}\nbal_acc test = {bal:.3f}", color=col, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(True)

fig.suptitle("Matrices de confusion sur le test set (n=14)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig3_confusion.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig3_confusion.pdf", bbox_inches="tight")
plt.close()
print(f"✓ Figure 3 : {OUT / 'fig3_confusion.png'}")

# ════════════════════════════════════════════════════════════════
# Figure 4 — Parcimonie
# ════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8.5, 4.7))

methods_s = ["SGCCA + LDA\n(NB09)",
             "Coop OvR\n(NB11)",
             "Multi GROUPED\n(NB14a)",
             "Multi UNGROUPED\n(NB14c)"]
ge_vars  = [nb09_vars_ge,  nb11_vars_ge,  nb14a_vars_ge,
            sum(nb14c_vars_ge_per_class.values())]
cgh_vars = [nb09_vars_cgh, nb11_vars_cgh, nb14a_vars_cgh, 0]
x = np.arange(len(methods_s))
w = 0.36
ax.bar(x - w/2, ge_vars,  width=w, color="#3498DB", alpha=0.88,
       label="GE (sur 15 702)", edgecolor="black", linewidth=0.6)
ax.bar(x + w/2, cgh_vars, width=w, color="#E74C3C", alpha=0.88,
       label="CGH (sur 1 229)", edgecolor="black", linewidth=0.6)
for i, (g, c) in enumerate(zip(ge_vars, cgh_vars)):
    ax.text(i - w/2, g + 1.5, str(g), ha="center", fontsize=10, fontweight="bold")
    ax.text(i + w/2, c + 1.5, str(c), ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(methods_s, fontsize=9)
ax.set_ylabel("Nombre de variables retenues (somme sur classes)")
ax.set_title("Parcimonie effective des quatre méthodes", pad=10)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.set_ylim(0, max(ge_vars) + 15)
plt.tight_layout()
plt.savefig(OUT / "fig4_sparsity.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig4_sparsity.pdf", bbox_inches="tight")
plt.close()
print(f"✓ Figure 4 : {OUT / 'fig4_sparsity.png'}")

# ════════════════════════════════════════════════════════════════
# Figure 5 — Recall par classe sur le test
# ════════════════════════════════════════════════════════════════

def per_class_recall(cm):
    rs = cm.sum(axis=1)
    return [cm[i, i] / rs[i] if rs[i] > 0 else 0 for i in range(3)]

recalls = {
    "SGCCA + LDA (NB09)":         per_class_recall(nb09_cm),
    "Cooperative OvR (NB11)":     per_class_recall(nb11_cm),
    "Multi. GROUPED (NB14a)":     per_class_recall(nb14a_cm),
    "Multi. UNGROUPED (NB14c)":   per_class_recall(nb14c_cm),
}
classes = ["cort", "dipg", "midl"]
cols    = [COL_SGCCA, COL_COOP, COL_MULTI_G, COL_MULTI_U]
x = np.arange(len(classes))
w = 0.20
offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

fig, ax = plt.subplots(figsize=(10, 5))
for off, col, (name, vals) in zip(offsets, cols, recalls.items()):
    ax.bar(x + off, vals, width=w, color=col, alpha=0.88,
           edgecolor="black", linewidth=0.6, label=name)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=11)
ax.set_ylabel("Recall (sensibilité) sur le test")
ax.set_title("Recall par classe", pad=10)
ax.set_ylim(0, 1.18)
ax.legend(loc="upper left", fontsize=8.5, ncol=2)
ax.grid(axis="y", alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig(OUT / "fig5_recall_per_class.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig5_recall_per_class.pdf", bbox_inches="tight")
plt.close()
print(f"✓ Figure 5 : {OUT / 'fig5_recall_per_class.png'}")

# ════════════════════════════════════════════════════════════════
# Figure 6 — Probabilités prédites sur les 3 vrais midl du test
# Layout : pour chaque patient (3 axes), 3 groupes de bars
# (un groupe = une méthode), chaque groupe contient 3 barres
# (P_cort, P_dipg, P_midl). Légende propre pour les méthodes,
# couleur pour les classes.
# ════════════════════════════════════════════════════════════════

probs_midl = {
    "NB11 Coop OvR":
        {"P11": (0.020, 0.100, 0.205),
         "P14": (0.040, 0.180, 0.205),
         "P19": (0.030, 0.080, 0.205)},
    "NB14a Multi GROUPED":
        {"P11": (0.027, 0.529, 0.444),
         "P14": (0.169, 0.506, 0.325),
         "P19": (0.017, 0.786, 0.197)},
    "NB14c Multi UNGROUPED":
        {"P11": (0.064, 0.818, 0.118),
         "P14": (0.134, 0.522, 0.344),
         "P19": (0.010, 0.816, 0.174)},
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
patients   = ["P11", "P14", "P19"]
methods_p  = list(probs_midl.keys())
classes_p  = ["cort", "dipg", "midl"]
class_cols = ["#5DADE2", "#F39C12", "#C0392B"]

bar_w        = 0.25
intra_gap    = 0.05   # gap entre barres d'un même groupe
group_w      = 3 * bar_w + 2 * intra_gap
inter_gap    = 0.55   # gap entre groupes de méthodes

for ax, pat in zip(axes, patients):
    group_centers = []
    for j, method in enumerate(methods_p):
        vals = probs_midl[method][pat]
        base = j * (group_w + inter_gap)
        positions = [base + i * (bar_w + intra_gap) for i in range(3)]
        ax.bar(positions, vals, width=bar_w,
               color=class_cols, edgecolor="black", linewidth=0.5)
        for pos, v in zip(positions, vals):
            ax.text(pos, v + 0.005, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="black")
        group_centers.append(base + (group_w - bar_w) / 2)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(methods_p, fontsize=8.5)
    ax.set_title(f"{pat} (vrai = midl)", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

axes[0].set_ylabel(r"$\hat P(\text{classe}\mid x)$")

# Légende classes
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=c, label=cl) for c, cl in zip(class_cols, classes_p)]
fig.legend(handles=handles, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=10)

fig.suptitle("Probabilités prédites pour les 3 vrais midl du test", fontsize=12.5, y=1.02)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(OUT / "fig6_midl_probs.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig6_midl_probs.pdf", bbox_inches="tight")
plt.close()
print(f"✓ Figure 6 : {OUT / 'fig6_midl_probs.png'}")

print("\n6 figures régénérées dans synthesis/figures/")
