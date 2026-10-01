# ==========================================================================
# 08_rgcca_sgcca.R — Reproduit la section 4.2 du papier RGCCA (Girka et al.,
# JSS 2025, Tenenhaus et al.) sur le dataset glioma IGR.
#
# Le papier est écrit par Arthur Tenenhaus (le référent du projet).
# Le case study du papier utilise EXACTEMENT le même dataset que ce projet
# (53 patients, 3 localisations, GE 15702 + CGH 1229).
#
# Score à reproduire :  CV balanced accuracy = 0.826 ± 0.136
#                       Test accuracy        = 91.7%
# ==========================================================================

suppressPackageStartupMessages({
  library(RGCCA)
  library(caret)
  library(MASS)
})

set.seed(0)

# --------------------------------------------------------------------------
# 1. Chargement des données depuis les CSV existants
# --------------------------------------------------------------------------
DATA_DIR <- file.path(dirname(getwd()), "data")
if (!dir.exists(DATA_DIR)) {
  DATA_DIR <- "../data"   # fallback si exécuté depuis models/
}
LABEL_ORDER <- c("cort", "dipg", "midl")

read_block <- function(block, split) {
  path <- file.path(DATA_DIR, sprintf("ge_cgh_locIGR__multiblocks__%s__%s.csv", block, split))
  df <- read.csv(path, stringsAsFactors = FALSE)
  rownames(df) <- df$row_id
  df$row_id <- NULL
  # Conversion virgules françaises → points
  df[] <- lapply(df, function(c) as.numeric(gsub(",", ".", as.character(c))))
  as.matrix(df)
}

read_targets <- function(split) {
  path <- file.path(DATA_DIR, sprintf("ge_cgh_locIGR__multiblocks__y__%s.csv", split))
  df <- read.csv(path, stringsAsFactors = FALSE)
  rownames(df) <- df$row_id
  df$row_id <- NULL
  apply(df[, LABEL_ORDER], 1, function(row) LABEL_ORDER[which.max(row)])
}

X_GE_train  <- read_block("GE",  "train")
X_GE_test   <- read_block("GE",  "test")
X_CGH_train <- read_block("CGH", "train")
X_CGH_test  <- read_block("CGH", "test")
y_train     <- factor(read_targets("train"), levels = LABEL_ORDER)
y_test      <- factor(read_targets("test"),  levels = LABEL_ORDER)

# Imputation médiane (par train uniquement)
fill_median <- function(train, test) {
  med <- apply(train, 2, median, na.rm = TRUE)
  for (j in seq_len(ncol(train))) {
    train[is.na(train[, j]), j] <- med[j]
    test[is.na(test[, j]), j]   <- med[j]
  }
  list(train = train, test = test)
}
ge  <- fill_median(X_GE_train,  X_GE_test);  X_GE_train  <- ge$train;  X_GE_test  <- ge$test
cgh <- fill_median(X_CGH_train, X_CGH_test); X_CGH_train <- cgh$train; X_CGH_test <- cgh$test

cat(sprintf("Train : %d   Test : %d\n", length(y_train), length(y_test)))
cat(sprintf("GE  : %d features   CGH : %d features\n",
            ncol(X_GE_train), ncol(X_CGH_train)))
cat("Train class distribution :\n"); print(table(y_train))
cat("Test  class distribution :\n"); print(table(y_test))

# --------------------------------------------------------------------------
# 2. Construction des blocs au format RGCCA
#    Le 3e bloc est la cible (factor → one-hot encoding implicite par RGCCA)
# --------------------------------------------------------------------------
blocks_train <- list(
  GE  = X_GE_train,
  CGH = X_CGH_train,
  y   = y_train
)
blocks_test <- list(
  GE  = X_GE_test,
  CGH = X_CGH_test,
  y   = y_test
)

# --------------------------------------------------------------------------
# 3. Pré-vol : RGCCA simple (régularisé) avec tau optimal Schäfer-Strimmer
# --------------------------------------------------------------------------
cat("\n=== RGCCA (régularisé, tau optimal) ===\n")
fit_rgcca <- rgcca(blocks = blocks_train, response = 3, tau = "optimal")
cat(sprintf("Tau optimaux : GE=%.4f  CGH=%.4f  y=%.4f\n",
            fit_rgcca$call$tau[1], fit_rgcca$call$tau[2], fit_rgcca$call$tau[3]))
cat(sprintf("Connection matrix automatique (response=3) :\n"))
print(fit_rgcca$call$connection)

# --------------------------------------------------------------------------
# 4. SGCCA — la version sparse, c'est elle qui fait le boulot en p >> n
# --------------------------------------------------------------------------
cat("\n=== SGCCA — recherche d'HP par CV (rgcca_cv) ===\n")
cat("Cette étape peut prendre 5-15 min selon la machine.\n")

t0 <- Sys.time()
cv_out <- rgcca_cv(
  blocks            = blocks_train,
  response          = 3,
  par_type          = "sparsity",
  par_value         = c(0.2, 0.2, 0),     # bornes sup pour la grille (paper)
  par_length        = 10,                  # 10 valeurs uniformément espacées
  prediction_model  = "lda",               # LDA sur les composantes (paper)
  validation        = "kfold",
  k                 = 7,                   # 7-fold CV (paper)
  n_run             = 3,                   # 3 répétitions (paper)
  metric            = "Balanced_Accuracy",
  n_cores           = 1
)
elapsed <- difftime(Sys.time(), t0, units = "mins")
cat(sprintf("Durée rgcca_cv : %.1f min\n", as.numeric(elapsed)))

print(summary(cv_out))

# --------------------------------------------------------------------------
# 5. Refit final avec les sparsity optimaux
# --------------------------------------------------------------------------
cat("\n=== Refit final SGCCA + LDA ===\n")
fit_final <- rgcca(cv_out)
print(summary(fit_final))

cat(sprintf("Sparsity finaux : GE=%.4f  CGH=%.4f\n",
            fit_final$call$sparsity[1], fit_final$call$sparsity[2]))
cat(sprintf("Variables sélectionnées : GE=%d/%d  CGH=%d/%d\n",
            sum(fit_final$a$GE  != 0), nrow(fit_final$a$GE),
            sum(fit_final$a$CGH != 0), nrow(fit_final$a$CGH)))

# --------------------------------------------------------------------------
# 6. Prédictions sur le test set
# --------------------------------------------------------------------------
cat("\n=== Prédictions test set ===\n")
pred <- rgcca_predict(fit_final, blocks_test = blocks_test, prediction_model = "lda")
cat("Confusion matrix (test):\n")
print(pred$confusion$test)

# Métriques détaillées via caret
cm <- confusionMatrix(
  factor(pred$class$test, levels = LABEL_ORDER),
  reference = y_test
)
cat("\nMétriques par classe (test set):\n")
print(cm$byClass[, c("Sensitivity", "Specificity", "Balanced Accuracy")])

cat(sprintf("\n>>> Test accuracy            : %.3f\n", cm$overall["Accuracy"]))
cat(sprintf(">>> Test balanced accuracy    : %.3f\n",
            mean(cm$byClass[, "Balanced Accuracy"])))
cat(sprintf(">>> CV balanced accuracy (7-fold x 3 runs) : %.3f ± %.3f\n",
            cv_out$bestTuneScore, cv_out$bestTuneSd))

# --------------------------------------------------------------------------
# 7. Stabilité (VIP-bootstrap) — quels gènes / régions sont stables
# --------------------------------------------------------------------------
cat("\n=== Stabilité par VIP-bootstrap (rgcca_stability) ===\n")
cat("Cette étape peut prendre 5-10 min.\n")

t0 <- Sys.time()
fit_stable <- rgcca_stability(
  fit_final,
  keep   = vapply(fit_final$a, function(x) mean(x != 0), FUN.VALUE = 1.0),
  n_boot = 100
)
cat(sprintf("Durée rgcca_stability : %.1f min\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

# --------------------------------------------------------------------------
# 8. Bootstrap pour les IC sur les poids
# --------------------------------------------------------------------------
cat("\n=== Bootstrap CI (rgcca_bootstrap) ===\n")
t0 <- Sys.time()
boot_out <- rgcca_bootstrap(fit_stable, n_boot = 500, n_cores = 1)
cat(sprintf("Durée rgcca_bootstrap : %.1f min\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

# Top variables GE par poids absolu (avec significativité)
boot_df <- summary(boot_out)
cat("\nTop 20 variables GE par bootstrap ratio :\n")
ge_boot <- boot_df[boot_df$block == "GE", ]
ge_boot <- ge_boot[order(-abs(ge_boot$bootstrap_ratio)), ]
print(head(ge_boot[, c("var", "estimate", "lower_bound", "upper_bound",
                       "bootstrap_ratio", "adjust.pval")], 20))

cat("\nTop 10 variables CGH par bootstrap ratio :\n")
cgh_boot <- boot_df[boot_df$block == "CGH", ]
cgh_boot <- cgh_boot[order(-abs(cgh_boot$bootstrap_ratio)), ]
print(head(cgh_boot[, c("var", "estimate", "lower_bound", "upper_bound",
                        "bootstrap_ratio", "adjust.pval")], 10))

# --------------------------------------------------------------------------
# 9. Sauvegardes
# --------------------------------------------------------------------------
out_dir <- file.path(dirname(getwd()), "exports")
if (!dir.exists(out_dir)) out_dir <- "../exports"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

saveRDS(fit_final,  file.path(out_dir, "08_sgcca_fit_final.rds"))
saveRDS(cv_out,     file.path(out_dir, "08_sgcca_cv.rds"))
saveRDS(fit_stable, file.path(out_dir, "08_sgcca_stable.rds"))
saveRDS(boot_out,   file.path(out_dir, "08_sgcca_bootstrap.rds"))

# Plots PDF
pdf(file.path(out_dir, "08_sgcca_plots.pdf"), width = 10, height = 6)
plot(cv_out, cex = 1.5)
plot(fit_final, type = "ave", cex = 0.7)
plot(fit_final, type = "sample", block = 1:2, comp = 1, resp = y_train, repel = TRUE, cex = 1.5)
plot(boot_out, block = 1, n_mark = 50, display_order = FALSE, cex = 1.2, show_star = TRUE)
plot(boot_out, block = 2, n_mark = 30, display_order = FALSE, cex = 1.2, show_star = TRUE)
dev.off()

cat("\n=== TERMINÉ ===\n")
cat(sprintf("Modèle final          : %s\n", file.path(out_dir, "08_sgcca_fit_final.rds")))
cat(sprintf("Plots                 : %s\n", file.path(out_dir, "08_sgcca_plots.pdf")))
