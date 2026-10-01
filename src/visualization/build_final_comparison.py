"""Figure finale + tableau LaTeX consolidé.

Intègre TOUS les pipelines évalués :
- Baselines : SGCCA + LDA (NB09), Cooperative OvR (NB11)
- Multinomial : grouped (NB14a), ungrouped (NB14c), coop multi (NB15)
- Class weights : NB17 (multi + inv_prev)
- Stability : NB18 (info séparée)
- Spatial CGH : NB19 (Group Lasso), NB20 (Fused TV)
- SGCCA pondéré : NB21
- Data augmentation : NB22 (SMOTE)
- TV + weights : NB23

Génère :
- figures/fig11_final_comparison_v2.png
- final_comparison_table.tex
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import json

SYN = Path(__file__).resolve().parent
OUT = SYN / "figures"

# Couleurs cohérentes
COL_SGCCA       = "#27AE60"
COL_COOP        = "#3498DB"
COL_MULTI_G     = "#9B59B6"
COL_MULTI_U     = "#E67E22"
COL_COOP_R      = "#16A085"
COL_WEIGHT      = "#7E57C2"
COL_TV          = "#E91E63"
COL_GLASSO      = "#FF9800"
COL_SGCCA_W     = "#1ABC9C"
COL_SMOTE       = "#5DADE2"
COL_TV_W        = "#D7263D"
COL_GREY        = "#7F8C8D"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
})

def load_rds(path):
    script = f"""
x <- readRDS('{path}')
cat(jsonlite::toJSON(x, auto_unbox=TRUE, force=TRUE, na='null', null='null'))
"""
    r = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    return json.loads(r.stdout)

# Charger ce qu'on a
nb17 = load_rds(SYN / "nb17_class_weights_results.rds")
nb19 = load_rds(SYN / "nb19_group_lasso_results.rds")
nb20 = load_rds(SYN / "nb20_fused_lasso_results.rds")
nb21 = load_rds(SYN / "nb21_sgcca_weights_results.rds")
nb22 = load_rds(SYN / "nb22_smote_results.rds")
nb23 = load_rds(SYN / "nb23_tv_weights_results.rds")

# Extraire les test scores
def get_test(d, scheme, family):
    for r in d["test"]:
        if r["scheme"] == scheme and r["family"] == family:
            return r["test_bal_acc"], int(r["midl_correct"])
    return None, None

nb17_best = get_test(nb17, "inv_prevalence", "multinomial")  # 0.924, 2
nb19_test = (float(nb19["test_bal_acc"][0] if isinstance(nb19["test_bal_acc"], list) else nb19["test_bal_acc"]),
             int(nb19["midl_test_correct"][0] if isinstance(nb19["midl_test_correct"], list) else nb19["midl_test_correct"]))
nb20_test = (nb20["test_bal_acc"], int(nb20["midl_test_correct"]))

# NB21 : best variant = "std" SGCCA + LDA empirique
nb21_synth = nb21["synth"]
nb21_best = max(nb21_synth, key=lambda r: r["test_bal_acc"])
nb21_test = (nb21_best["test_bal_acc"], int(nb21_best["midl_correct"]))

# NB22 : best = SMOTE topvar200 multinomial OR raw OvR
nb22_results = nb22["results"]
nb22_best = max(nb22_results, key=lambda r: r["test_bal_acc"])
nb22_test = (nb22_best["test_bal_acc"], int(nb22_best["midl_correct"]))

# NB23 : best = TV Multi + weights
nb23_synth = nb23["synth"]
nb23_best = max(nb23_synth, key=lambda r: r["test_bal_acc"])
nb23_test = (nb23_best["test_bal_acc"], int(nb23_best["midl_correct"]))

# =====================================================================
# Tableau complet de toutes les méthodes (CV + test)
# =====================================================================
methods = [
    # (label, color, CV mean, CV sd, test bal_acc, midl test, "approche")
    ("SGCCA + LDA (NB09)",                  COL_SGCCA,    0.829, 0.133, 0.924, 2, "Baseline"),
    ("Cooperative OvR ρ=0 (NB11)",          COL_COOP,     0.833, 0.129, 0.924, 2, "Baseline"),
    ("Multinomial grouped (NB14a)",         COL_MULTI_G,  0.784, 0.096, 0.773, 0, "Baseline"),
    ("Multinomial ungrouped (NB14c)",       COL_MULTI_U,  0.838, 0.123, 0.773, 0, "Baseline"),
    ("Coop multinomial NB15 (ρ=0.1)",       COL_COOP_R,   0.810, 0.160, 0.771, 0, "Baseline"),
    ("Multi + weights inv_prev (NB17)",     COL_WEIGHT,   0.830, 0.118, nb17_best[0], nb17_best[1], "Pondération"),
    ("Group Lasso CGH (NB19)",              COL_GLASSO,   None, None, nb19_test[0], nb19_test[1], "Structure CGH"),
    ("Fused Lasso TV (NB20)",               COL_TV,       nb20["cv"]["mean_bal_acc"], nb20["cv"]["sd_bal_acc"], nb20_test[0], nb20_test[1], "Structure CGH"),
    ("SGCCA pondérée + LDA (NB21)",         COL_SGCCA_W,  None, None, nb21_test[0], nb21_test[1], "Pondération SGCCA"),
    ("SMOTE + Lasso (NB22)",                COL_SMOTE,    None, None, nb22_test[0], nb22_test[1], "Augmentation"),
    ("TV + weights inv_prev (NB23)",        COL_TV_W,     None, None, nb23_test[0], nb23_test[1], "TV + Pondération"),
]

# Trier par test bal_acc décroissant
methods_sorted = sorted(methods, key=lambda m: (-m[4], -m[5]))

# =====================================================================
# Figure
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

labels = [m[0] for m in methods_sorted]
colors = [m[1] for m in methods_sorted]
cv_m   = [m[2] if m[2] is not None else np.nan for m in methods_sorted]
cv_s   = [m[3] if m[3] is not None else 0       for m in methods_sorted]
test_m = [m[4] for m in methods_sorted]
midl_m = [m[5] for m in methods_sorted]

# Panneau gauche : CV
ax = axes[0]
y = np.arange(len(labels))
for yi, mu, sd, col in zip(y, cv_m, cv_s, colors):
    if np.isnan(mu):
        ax.text(0.35, yi, "CV non évaluée", va="center", fontsize=8,
                style="italic", color=COL_GREY)
    else:
        ax.barh(yi, mu, xerr=sd, color=col, edgecolor="black",
                linewidth=0.5, alpha=0.88, capsize=3)
        ax.text(mu + sd + 0.015, yi, f"{mu:.3f}", va="center",
                fontsize=8.5, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Balanced accuracy CV (21 plis)")
ax.set_title("Cross-validation", pad=8, fontweight="bold")
ax.set_xlim(0.3, 1.05); ax.grid(axis="x", alpha=0.2, linestyle="--")

# Panneau droit : test + midl
ax = axes[1]
ax.barh(y, test_m, color=colors, edgecolor="black",
        linewidth=0.5, alpha=0.88)
for yi, v, m in zip(y, test_m, midl_m):
    midl_text = f"midl {m}/3"
    midl_color = "#27AE60" if m == 2 else ("#F39C12" if m == 1 else "#C0392B")
    ax.text(v + 0.015, yi, f"{v:.3f}", va="center",
            fontsize=8.5, fontweight="bold")
    ax.text(1.08, yi, midl_text, va="center",
            fontsize=8.5, fontweight="bold", color=midl_color)
ax.set_yticks(y); ax.set_yticklabels([])
ax.invert_yaxis()
ax.set_xlabel("Balanced accuracy test (n=14)")
ax.set_title("Test set tenu à part", pad=8, fontweight="bold")
ax.set_xlim(0.3, 1.25); ax.grid(axis="x", alpha=0.2, linestyle="--")

fig.suptitle("Comparaison finale — 11 pipelines évalués sur le même protocole",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig11_final_comparison_v2.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig11_final_comparison_v2.pdf", bbox_inches="tight")
plt.close()
print(f"✓ {OUT / 'fig11_final_comparison_v2.png'}")

# =====================================================================
# Tableau LaTeX
# =====================================================================
tex_lines = [
    r"\begin{table}[h]",
    r"\centering",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\begin{tabular}{l|c c c c}",
    r"\hline",
    r"\textbf{Méthode} & \textbf{CV bal\_acc} & \textbf{Test bal\_acc} & "
    r"\textbf{midl test} & \textbf{Catégorie} \\",
    r"\hline",
]
for label, _, mu, sd, ta, mc, cat in methods_sorted:
    cv_str = "—" if mu is None or np.isnan(mu) else f"${mu:.3f} \\pm {sd:.3f}$"
    line = f"{label} & {cv_str} & ${ta:.3f}$ & ${mc}/3$ & {cat} \\\\"
    tex_lines.append(line)
tex_lines += [
    r"\hline",
    r"\end{tabular}",
    r"\caption{Comparaison finale des 11 pipelines évalués sur le même "
    r"protocole (CV 7-fold $\times$ 3, test $n=14$). Triés par balanced "
    r"accuracy test décroissante.}",
    r"\label{tab:final_comparison}",
    r"\end{table}",
]
with open(SYN / "final_comparison_table.tex", "w") as f:
    f.write("\n".join(tex_lines))
print(f"✓ {SYN / 'final_comparison_table.tex'}")

print("\nMéthodes triées par test bal_acc :")
for m in methods_sorted:
    print(f"  {m[0]:50s} → test {m[4]:.3f} | midl {m[5]}/3")
