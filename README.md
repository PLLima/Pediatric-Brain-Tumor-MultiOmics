# Pediatric-Brain-Tumor-MultiOmics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python/R](https://img.shields.io/badge/Language-Python%20%7C%20R-blue)]()

An interpretable, multi-omics AI model designed to predict the anatomical localization of pediatric brain tumors—specifically distinguishing cortical, midline, and Diffuse Intrinsic Pontine Glioma (DIPG) cases. Accurate localization mapping via genetic profiles is a critical step towards understanding molecular signatures and developing highly targeted therapeutic interventions.

---

## 🎯 Overview

Pediatric brain tumors, particularly DIPG, present profound clinical challenges. The location of the tumor dictates resectability, radiotherapy protocols, and overall prognosis. This project builds a machine learning pipeline capable of classifying tumor localization directly from molecular data, circumventing reliance on pure imaging and potentially identifying targetable biological mechanisms.

### Main Challenges Addressed:
- **$p \gg n$ Regime:** Operating in a high-dimensional space with 39 training patients against $\simeq$ 17,000 biological features.
- **Extreme Class Imbalance:** The *midline* (`midl`) class is significantly underrepresented ($\sim$20%), demanding careful loss re-weighting and metric selection.
- **Block Dominance:** Gene Expression (GE) data heavily outweighs Chromosomal Alteration (CGH) data in dimensionality, requiring sophisticated fusion methodologies rather than naïve concatenation.

---

## 🔬 Methodology

To navigate the high-dimensional, unbalanced, multi-modal data, this project explores and rigorously evaluates three distinct modeling pipelines:

1. **Sparse Generalized Canonical Correlation Analysis (SGCCA) + LDA** *(Most Robust)*
   SGCCA provides supervised dimensionality reduction, intrinsically selecting a sparse subset of features. It predominantly isolates 68 features from the GE block while strictly constraining the non-informative CGH block. A Linear Discriminant Analysis (LDA) classifier is subsequently trained on these reduced components.
2. **Cooperative Learning (`cv.multiview`)**
   A state-of-the-art approach attempting to structurally enforce alignment between the multi-omic blocks. Empirical results demonstrated algorithmic instability when forcing block agreement ($\rho > 0$), resulting in the strategy gracefully degrading into an effective One-vs-Rest (OvR) Lasso model.
3. **Multinomial Logistic Regression (Elastic-Net)**
   Penalized multi-class logistic regression. Evaluation revealed that varying the sparsity constraint structure (grouped vs. ungrouped penalties) heavily influences the resulting genetic interpretations.

---

## 📊 Key Results

- **Feature Importance:** The predictive signal is overwhelmingly driven by the Gene Expression (GE) block.
- **Performance:** The SGCCA + LDA pipeline successfully mitigates the class imbalance and reliably recovers the rare `midl` class, yielding a robust cross-validated balanced accuracy of **0.829**.
- **Interpretability:** The sparse regularizations yield actionable, finite lists of discriminative genes, directly contributing to biological interpretability.

Detailed results and visualizations are strictly maintained in the [`reports/`](./reports) directory.

---

## 📁 Repository Structure

```text
├── data/                  # Ignored by git; contains train/test splits (clinical, GE, CGH, labels)
├── notebooks/             # Jupyter notebooks and R scripts for EDA, modeling, and evaluation
├── reports/               # Final documentation, PDF reports, and synthesized markdown summaries
│   ├── figures/           # Auto-generated plots, confusion matrices, and EDA charts
│   ├── rapport.pdf        # Comprehensive theoretical and statistical foundation
│   └── pipeline_cooperative.pdf
├── src/
│   └── visualization/     # Source code containing python scripts to generate reporting figures
├── README.md              # Project overview
└── .ai_context.md         # Synchronized context reference for AI contributors
```

---

## 🚀 Installation & Usage

### Prerequisites
The environment requires a dual setup capable of running both **R** and **Python** code.
- R version $\geq$ 4.0 (with `glmnet`, `RGCCA`, `caret`, `multiview`)
- Python version $\geq$ 3.8 (with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`)

### Setup
Clone the repository:
```bash
git clone https://github.com/PLLima/Pediatric-Brain-Tumor-MultiOmics.git
cd Pediatric-Brain-Tumor-MultiOmics
```

*Note: Ensure your `data/` directory is populated with the correct datasets (e.g., `cgh_train.csv`, `ge_train.csv`, `labels_train.csv`) prior to running the pipelines, as data is excluded from version control to maintain privacy and repository health.*

### Execution
Modeling pipelines can be executed via the corresponding notebooks or R scripts located in the `notebooks/` directory.

To regenerate the synthesis figures, execute the visualization scripts:
```bash
python src/visualization/build_figures.py
```

---

## 📜 License & Citation

This project is licensed under the [MIT License](LICENSE). While this permits permissive usage including commercial utilization, any academic or research adaptation must appropriately cite the original authors and the CentraleSupélec data science framework from which this methodology originated.
