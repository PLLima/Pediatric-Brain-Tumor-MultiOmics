# =====================================================================
# NB14c — Multinomial Lasso UNGROUPED
# =====================================================================
# Variante de NB14 avec `type.multinomial = "ungrouped"` au lieu de
# "grouped". Sert à discriminer DEUX hypothèses concurrentes pour
# expliquer pourquoi NB14a (multinomial grouped) recall midl 0/3 alors
# que NB11 (OvR binomial) recall midl 2/3 :
#
#   H1 — La group lasso force midl à partager les mêmes features que
#        cort/dipg, ce qui dilue son signal et l'empêche d'avoir un
#        modèle null calibré sur la prévalence.
#   H2 — Le softmax multinomial est intrinsèquement inadapté aux
#        classes rares (couplage des probabilités, pas de plancher
#        de prévalence à la OvR-intercept).
#
# Lecture du résultat :
#   delta_bal_acc = bal_acc(ungrouped) - bal_acc(grouped)
#   - delta > +0.03 ou midl_test >= 1/3   → H1 vraie (group lasso coupable)
#   - delta ≈ 0  et midl_test = 0/3       → H2 vraie (softmax intrinsèquement KO)
#
# Usage standalone (depuis la console R) :
#   setwd("/Users/ruben/Documents/Brain-Cancer-Prediction-Model/models")
#   source("14c_multinomial_ungrouped.R", encoding = "UTF-8")
#
# Usage depuis NB14 (après avoir exécuté §1 §2) :
#   source("14c_multinomial_ungrouped.R")
# =====================================================================

# ---------- 0. Packages ----------
suppressPackageStartupMessages({
  required_packages <- c("glmnet", "data.table", "caret")
  to_install <- required_packages[!vapply(required_packages, requireNamespace,
                                          logical(1), quietly = TRUE)]
  if (length(to_install) > 0) {
    install.packages(to_install, repos = "https://cloud.r-project.org")
  }
  library(glmnet)
  library(data.table)
  library(caret)
})

# ---------- 1. Constantes (créées si manquantes) ----------
if (!exists("SEED"))        SEED        <- 42
if (!exists("LABEL_ORDER")) LABEL_ORDER <- c("cort", "dipg", "midl")
set.seed(SEED)

# Valeurs de référence NB14a (multinomial grouped) — documentées
# dans le notebook NB14, cellule 7f18ac9f.
if (!exists("nb14_cv_mean")) nb14_cv_mean <- 0.784
if (!exists("nb14_cv_sd"))   nb14_cv_sd   <- 0.096

# ---------- 2. Chargement des données si pas déjà en mémoire ----------
needs_data <- !all(vapply(c("X_train","X_test","y_train","y_test"),
                          exists, logical(1)))

if (needs_data) {
  cat("Variables X_train/y_train absentes — chargement des données…\n")

  # Localiser data/ — fonctionne depuis models/ ou racine projet
  root <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  candidates <- c(file.path(root,            "data"),
                  file.path(dirname(root),   "data"),
                  file.path(root, "Brain-Cancer-Prediction-Model", "data"))
  data_dir <- candidates[dir.exists(candidates)][1]
  if (is.na(data_dir)) stop("Impossible de trouver le dossier data/. ",
                            "setwd() vers le dossier models/ du projet.")
  cat("  data_dir = ", data_dir, "\n")

  to_numeric_frame <- function(df) {
    rn <- rownames(df)
    out <- as.data.frame(
      lapply(df, function(x) as.numeric(gsub(",", ".", as.character(x), fixed = TRUE))),
      check.names = FALSE
    )
    rownames(out) <- rn
    out
  }
  extract_id_column <- function(df) if ("row_id" %in% names(df)) "row_id" else names(df)[1]

  load_block <- function(block_name, split) {
    path <- file.path(data_dir,
      sprintf("ge_cgh_locIGR__multiblocks__%s__%s.csv", block_name, split))
    df <- as.data.frame(data.table::fread(path, check.names = FALSE))
    id_col <- extract_id_column(df)
    rownames(df) <- as.character(df[[id_col]])
    df[[id_col]] <- NULL
    to_numeric_frame(df)
  }

  load_targets <- function(split) {
    path <- file.path(data_dir,
      sprintf("ge_cgh_locIGR__multiblocks__y__%s.csv", split))
    y_df <- as.data.frame(data.table::fread(path, check.names = FALSE))
    id_col <- extract_id_column(y_df)
    rownames(y_df) <- as.character(y_df[[id_col]])
    y_df[[id_col]] <- NULL
    factor(
      LABEL_ORDER[max.col(as.matrix(y_df[, LABEL_ORDER]), ties.method = "first")],
      levels = LABEL_ORDER
    ) -> targets
    names(targets) <- rownames(y_df)
    targets
  }

  fill_missing_from_train <- function(train_df, test_df) {
    medians <- vapply(train_df, median, numeric(1), na.rm = TRUE)
    for (col in names(train_df)) {
      train_df[[col]][is.na(train_df[[col]])] <- medians[[col]]
      test_df[[col]] [is.na(test_df[[col]])]  <- medians[[col]]
    }
    list(train = train_df, test = test_df)
  }

  X_ge_train  <- load_block("GE",  "train")
  X_ge_test   <- load_block("GE",  "test")
  X_cgh_train <- load_block("CGH", "train")
  X_cgh_test  <- load_block("CGH", "test")
  y_train     <- load_targets("train")
  y_test      <- load_targets("test")

  train_ids <- Reduce(intersect, list(rownames(X_ge_train), rownames(X_cgh_train),
                                       names(y_train)))
  test_ids  <- Reduce(intersect, list(rownames(X_ge_test),  rownames(X_cgh_test),
                                       names(y_test)))

  X_ge_train  <- as.matrix(X_ge_train [train_ids, , drop = FALSE])
  X_cgh_train <- as.matrix(X_cgh_train[train_ids, , drop = FALSE])
  y_train     <- y_train [train_ids]
  X_ge_test   <- as.matrix(X_ge_test  [test_ids,  , drop = FALSE])
  X_cgh_test  <- as.matrix(X_cgh_test [test_ids,  , drop = FALSE])
  y_test      <- y_test   [test_ids]

  filled_ge  <- fill_missing_from_train(as.data.frame(X_ge_train),
                                         as.data.frame(X_ge_test))
  X_ge_train <- as.matrix(filled_ge$train);  X_ge_test <- as.matrix(filled_ge$test)
  filled_cgh <- fill_missing_from_train(as.data.frame(X_cgh_train),
                                         as.data.frame(X_cgh_test))
  X_cgh_train <- as.matrix(filled_cgh$train); X_cgh_test <- as.matrix(filled_cgh$test)

  X_train <- cbind(X_ge_train, X_cgh_train)
  X_test  <- cbind(X_ge_test,  X_cgh_test)
  colnames(X_train)[1:ncol(X_ge_train)]                   <- paste0("GE__",  colnames(X_ge_train))
  colnames(X_train)[(ncol(X_ge_train)+1):ncol(X_train)]   <- paste0("CGH__", colnames(X_cgh_train))
  colnames(X_test) <- colnames(X_train)

  cat(sprintf("Train: %d patients | Test: %d patients\n",
              length(y_train), length(y_test)))
  cat(sprintf("X concaténé : %d × %d (GE: %d + CGH: %d)\n\n",
              nrow(X_train), ncol(X_train),
              ncol(X_ge_train), ncol(X_cgh_train)))
}

stopifnot(exists("X_train"), exists("y_train"),
          exists("X_test"),  exists("y_test"))

# ---------- 3. CV sweep ungrouped ----------
ALPHA_GRID <- c(0, 0.5, 1.0)
cv_results_ungrouped <- data.frame()

set.seed(SEED)
outer_folds_u <- caret::createMultiFolds(y_train, k = 7, times = 3)

for (alpha_val in ALPHA_GRID) {
  cat(sprintf("\n=== alpha = %.1f (ungrouped) ===\n", alpha_val))
  fold_scores  <- numeric(length(outer_folds_u))
  midl_recalls <- numeric(length(outer_folds_u))
  t0 <- Sys.time()

  for (i in seq_along(outer_folds_u)) {
    tr_idx <- outer_folds_u[[i]]
    va_idx <- setdiff(seq_along(y_train), tr_idx)

    fit_fold <- tryCatch(
      cv.glmnet(x = X_train[tr_idx, , drop = FALSE],
                y = y_train[tr_idx],
                family           = "multinomial",
                type.multinomial = "ungrouped",   # ← seul changement vs NB14a
                alpha            = alpha_val,
                nfolds           = 5,
                standardize      = TRUE),
      error = function(e) { message("  fold ", i, " failed: ", conditionMessage(e)); NULL }
    )
    if (is.null(fit_fold)) { fold_scores[i] <- NA; midl_recalls[i] <- NA; next }

    pred_va <- predict(fit_fold,
                       newx = X_train[va_idx, , drop = FALSE],
                       s = "lambda.min", type = "class")[, 1]
    cm <- caret::confusionMatrix(
      factor(pred_va,           levels = LABEL_ORDER),
      factor(y_train[va_idx],   levels = LABEL_ORDER)
    )
    fold_scores[i]  <- mean(cm$byClass[, "Balanced Accuracy"], na.rm = TRUE)
    midl_recalls[i] <- cm$byClass["Class: midl", "Sensitivity"]
  }

  cv_results_ungrouped <- rbind(cv_results_ungrouped, data.frame(
    alpha            = alpha_val,
    mean_bal_acc     = mean(fold_scores,  na.rm = TRUE),
    sd_bal_acc       = sd  (fold_scores,  na.rm = TRUE),
    mean_midl_recall = mean(midl_recalls, na.rm = TRUE),
    sd_midl_recall   = sd  (midl_recalls, na.rm = TRUE),
    n_folds          = sum(!is.na(fold_scores))
  ))
  cat(sprintf("  Bal_acc moy = %.3f ± %.3f | Recall midl = %.3f ± %.3f  (%.1f min)\n",
              tail(cv_results_ungrouped$mean_bal_acc, 1),
              tail(cv_results_ungrouped$sd_bal_acc, 1),
              tail(cv_results_ungrouped$mean_midl_recall, 1),
              tail(cv_results_ungrouped$sd_midl_recall, 1),
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}

cat("\n========== CV ungrouped ==========\n")
print(cv_results_ungrouped)

best_idx_u   <- which.max(cv_results_ungrouped$mean_bal_acc)
best_alpha_u <- cv_results_ungrouped$alpha[best_idx_u]
cat(sprintf("\n>>> Meilleur alpha (ungrouped) : %.1f  (CV bal_acc = %.3f ± %.3f)\n",
            best_alpha_u,
            cv_results_ungrouped$mean_bal_acc[best_idx_u],
            cv_results_ungrouped$sd_bal_acc[best_idx_u]))

# ---------- 4. Refit final ungrouped + évaluation test ----------
set.seed(SEED)
fit_final_u <- cv.glmnet(
  x = X_train, y = y_train,
  family           = "multinomial",
  type.multinomial = "ungrouped",
  alpha            = best_alpha_u,
  nfolds           = 10,
  standardize      = TRUE
)

cat(sprintf("\nRefit ungrouped : alpha=%.1f, lambda.min=%.5f\n",
            best_alpha_u, fit_final_u$lambda.min))

coefs_u <- coef(fit_final_u, s = "lambda.min")
cat("\nVariables non-nulles par classe (ungrouped — peuvent différer) :\n")
n_per_class <- list()
for (cl in names(coefs_u)) {
  cmat <- coefs_u[[cl]]
  feat_names <- rownames(cmat)[2:nrow(cmat)]
  beta       <- cmat[2:nrow(cmat), 1]
  intercept  <- cmat[1, 1]
  is_ge  <- grepl("^GE__",  feat_names)
  is_cgh <- grepl("^CGH__", feat_names)
  nz <- abs(beta) > 1e-8
  cat(sprintf("  %s : %d GE + %d CGH = %d variables  (intercept = %+.3f)\n",
              cl, sum(nz & is_ge), sum(nz & is_cgh), sum(nz), intercept))
  n_per_class[[cl]] <- sum(nz)
}

pred_test_u  <- predict(fit_final_u, newx = X_test, s = "lambda.min", type = "class")[, 1]
probs_test_u <- predict(fit_final_u, newx = X_test, s = "lambda.min", type = "response")[, , 1]

cat("\nProbabilités prédites test (ungrouped) :\n")
print(round(probs_test_u, 3))

cm_u <- caret::confusionMatrix(
  factor(pred_test_u, levels = LABEL_ORDER),
  factor(y_test,      levels = LABEL_ORDER)
)
cat("\n=== Matrice de confusion test (ungrouped) ===\n")
print(cm_u$table)
cat(sprintf("\nAccuracy : %.3f | Balanced accuracy : %.3f\n",
            cm_u$overall["Accuracy"],
            mean(cm_u$byClass[, "Balanced Accuracy"])))
cat("\nBal_acc par classe :\n")
print(round(cm_u$byClass[, "Balanced Accuracy"], 3))

# ---------- 5. Discrimination H1 vs H2 ----------
midl_test_recovered <- cm_u$table["midl", "midl"]
delta_bal_acc       <- cv_results_ungrouped$mean_bal_acc[best_idx_u] - nb14_cv_mean

cat("\n========== TEST D'HYPOTHÈSE ==========\n")
cat(sprintf("Grouped   (NB14a) : CV bal_acc = %.3f ± %.3f | midl test = 0/3\n",
            nb14_cv_mean, nb14_cv_sd))
cat(sprintf("Ungrouped (NB14c) : CV bal_acc = %.3f ± %.3f | midl test = %d/3\n",
            cv_results_ungrouped$mean_bal_acc[best_idx_u],
            cv_results_ungrouped$sd_bal_acc[best_idx_u],
            midl_test_recovered))
cat(sprintf("Δ ungrouped − grouped : %+.3f en bal_acc CV\n\n", delta_bal_acc))

if (delta_bal_acc > 0.03 || midl_test_recovered >= 1) {
  cat("→ H1 CONFIRMÉE : la group lasso était coupable.\n")
  cat("  Le couplage des features entre classes empêchait midl d'avoir\n")
  cat("  son propre modèle null calibré. En ungrouped, midl peut sélectionner\n")
  cat("  zéro feature (ou très peu) et laisser l'intercept de prévalence\n")
  cat("  jouer son rôle de plancher.\n")
  cat("  Implication rapport : la perte NB14 vs NB11 venait du `type.multinomial`,\n")
  cat("  pas du softmax. À documenter comme HP critique non exploré dans la v1.\n")
} else {
  cat("→ H2 CONFIRMÉE : ungrouped ne sauve pas midl.\n")
  cat("  Même avec la liberté de sélectionner ses propres features, midl\n")
  cat("  reste mal modélisé par le softmax multinomial. La normalisation\n")
  cat("  Σ_k P(c=k|x) = 1 redistribue la probabilité midl vers les classes\n")
  cat("  majoritaires dès qu'il y a la moindre incertitude.\n")
  cat("  Implication rapport : la formulation OvR + argmax + λ per-classe (NB11)\n")
  cat("  est structurellement supérieure pour les classes très déséquilibrées.\n")
  cat("  C'est un résultat publiable.\n")
}

# ---------- 6. Sauvegarde ----------
out_path <- file.path(getwd(), "nb14c_results.rds")
saveRDS(list(
  cv_results_grouped   = data.frame(alpha = ALPHA_GRID,
                                     mean_bal_acc = c(NA, NA, nb14_cv_mean),
                                     sd_bal_acc   = c(NA, NA, nb14_cv_sd)),
  cv_results_ungrouped = cv_results_ungrouped,
  best_alpha_u         = best_alpha_u,
  cm_u                 = cm_u$table,
  probs_test_u         = probs_test_u,
  n_per_class          = n_per_class,
  midl_test_recovered  = midl_test_recovered
), file = out_path)

cat(sprintf("\nRésultats sauvegardés : %s\n", out_path))
