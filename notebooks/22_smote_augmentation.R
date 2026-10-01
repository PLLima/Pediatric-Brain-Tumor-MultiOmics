# =====================================================================
# NB22 — Augmentation de données SMOTE pour la classe midl
# =====================================================================
# SMOTE (Chawla et al. 2002) génère des observations synthétiques pour
# la classe minoritaire en interpolant entre observations existantes
# et leurs k-NN.
#
# Difficulté en p >> n : l'interpolation dans 16931 dimensions n'a pas
# de garantie biologique. On teste donc deux variantes :
#
#   A. SMOTE direct sur l'espace des features (16931D)
#   B. SMOTE sur top-200 features par variance (espace réduit)
#
# Dans chaque cas, on entraîne :
#   - Multinomial Lasso ungrouped (NB14c-style)
#   - OvR binomial Lasso (NB11-style sans cooperative)
#
# Implémentation manuelle de SMOTE : pour chaque obs minoritaire, on tire
# un voisin k-NN au hasard parmi les minoritaires, et on génère un point
# x_new = x + alpha * (x_neighbor - x), alpha ~ U[0,1].
# =====================================================================

setwd("/Users/ruben/Documents/Brain-Cancer-Prediction-Model/models")
suppressPackageStartupMessages({
  if (!requireNamespace("FNN", quietly = TRUE))
    install.packages("FNN", repos = "https://cloud.r-project.org")
  library(glmnet); library(data.table); library(caret); library(FNN)
})
SEED <- 42; set.seed(SEED)
LABEL_ORDER <- c("cort", "dipg", "midl")
data_dir <- "../data"

to_numeric_frame <- function(df) {
  rn <- rownames(df)
  out <- as.data.frame(lapply(df, function(x) as.numeric(gsub(",",".",as.character(x),fixed=TRUE))),
                       check.names=FALSE); rownames(out) <- rn; out
}
load_block <- function(blk, split) {
  df <- as.data.frame(fread(file.path(data_dir,
    sprintf("ge_cgh_locIGR__multiblocks__%s__%s.csv", blk, split)),check.names=FALSE))
  rownames(df) <- as.character(df[[1]]); df[[1]] <- NULL; to_numeric_frame(df)
}
load_y <- function(split) {
  df <- as.data.frame(fread(file.path(data_dir,
    sprintf("ge_cgh_locIGR__multiblocks__y__%s.csv", split)),check.names=FALSE))
  ids <- as.character(df[[1]]); df[[1]] <- NULL
  y <- factor(LABEL_ORDER[max.col(as.matrix(df[, LABEL_ORDER]), ties.method="first")],
              levels=LABEL_ORDER); names(y) <- ids; y
}
GE_tr <- load_block("GE","train"); GE_te <- load_block("GE","test")
CGH_tr <- load_block("CGH","train"); CGH_te <- load_block("CGH","test")
y_tr_raw <- load_y("train"); y_te_raw <- load_y("test")
tr_ids <- Reduce(intersect, list(rownames(GE_tr), rownames(CGH_tr), names(y_tr_raw)))
te_ids <- Reduce(intersect, list(rownames(GE_te), rownames(CGH_te), names(y_te_raw)))
GE_tr <- as.matrix(GE_tr[tr_ids,]); CGH_tr <- as.matrix(CGH_tr[tr_ids,])
GE_te <- as.matrix(GE_te[te_ids,]); CGH_te <- as.matrix(CGH_te[te_ids,])
y_train <- y_tr_raw[tr_ids]; y_test <- y_te_raw[te_ids]
imp <- function(tr, te) { med <- apply(tr,2,median,na.rm=TRUE)
  for (j in 1:ncol(tr)) { tr[is.na(tr[,j]),j] <- med[j]; te[is.na(te[,j]),j] <- med[j] }
  list(tr=tr,te=te) }
f <- imp(GE_tr, GE_te); GE_tr <- f$tr; GE_te <- f$te
f <- imp(CGH_tr, CGH_te); CGH_tr <- f$tr; CGH_te <- f$te

X_train <- cbind(GE_tr, CGH_tr); X_test <- cbind(GE_te, CGH_te)
colnames(X_train)[1:ncol(GE_tr)] <- paste0("GE__", colnames(GE_tr))
colnames(X_train)[(ncol(GE_tr)+1):ncol(X_train)] <- paste0("CGH__", colnames(CGH_tr))
colnames(X_test) <- colnames(X_train)

# ---- SMOTE manuel pour multi-classe ----
smote_multi <- function(X, y, target_count = NULL, k = 5, seed = 42) {
  set.seed(seed)
  classes <- levels(y); n_c <- table(y)
  if (is.null(target_count)) target_count <- max(n_c)
  X_new <- X; y_new <- y
  for (cl in classes) {
    n_cur <- n_c[cl]; n_synth <- target_count - n_cur
    if (n_synth <= 0) next
    idx_cl <- which(y == cl)
    if (length(idx_cl) < 2) next
    X_cl <- X[idx_cl, , drop = FALSE]
    # k-NN au sein de la classe
    knn_idx <- FNN::get.knn(X_cl, k = min(k, length(idx_cl) - 1))$nn.index
    # Génération
    synth <- matrix(0, n_synth, ncol(X))
    for (s in 1:n_synth) {
      i <- sample(length(idx_cl), 1)
      j <- knn_idx[i, sample(ncol(knn_idx), 1)]
      alpha <- runif(1)
      synth[s, ] <- X_cl[i, ] + alpha * (X_cl[j, ] - X_cl[i, ])
    }
    colnames(synth) <- colnames(X)
    X_new <- rbind(X_new, synth)
    y_new <- c(as.character(y_new), rep(cl, n_synth))
  }
  list(X = X_new, y = factor(y_new, levels = classes))
}

# ---- Pipeline : SMOTE + fit + test ----
eval_smote <- function(X_tr, y_tr, X_te, y_te, smote_space,
                        target_count = 16, family_type) {
  set.seed(SEED)
  if (smote_space == "raw") {
    aug <- smote_multi(X_tr, y_tr, target_count = target_count)
  } else if (smote_space == "topvar200") {
    top_idx <- order(apply(X_tr, 2, var), decreasing = TRUE)[1:200]
    X_red <- X_tr[, top_idx]
    aug <- smote_multi(X_red, y_tr, target_count = target_count)
    # Reprojetter dans l'espace original : pour les obs synthétiques, prendre
    # les coordonnées sur top_idx et padder les autres avec une moyenne (proxy)
    X_aug_full <- matrix(0, nrow(aug$X), ncol(X_tr))
    colnames(X_aug_full) <- colnames(X_tr)
    X_aug_full[, top_idx] <- aug$X
    # Pour features hors top_idx : on copie les valeurs depuis l'obs originale
    # la plus proche (k-NN en espace réduit). Simplification : valeur median train.
    other_idx <- setdiff(seq_len(ncol(X_tr)), top_idx)
    med_other <- apply(X_tr[, other_idx], 2, median)
    X_aug_full[, other_idx] <- matrix(med_other,
                                       nrow = nrow(aug$X),
                                       ncol = length(other_idx),
                                       byrow = TRUE)
    # Les vraies obs gardent leurs valeurs
    n_orig <- nrow(X_tr)
    X_aug_full[1:n_orig, ] <- X_tr
    aug$X <- X_aug_full
  }
  # Fit
  if (family_type == "multinomial") {
    fit <- cv.glmnet(aug$X, aug$y, family = "multinomial",
                     type.multinomial = "ungrouped", alpha = 1,
                     nfolds = 5, standardize = TRUE)
    pred <- predict(fit, X_te, s = "lambda.min", type = "class")[, 1]
  } else if (family_type == "ovr") {
    classes <- levels(aug$y)
    probs <- matrix(0, nrow(X_te), length(classes)); colnames(probs) <- classes
    for (k in seq_along(classes)) {
      y_k <- as.numeric(aug$y == classes[k])
      fit_k <- cv.glmnet(aug$X, y_k, family = "binomial", alpha = 1,
                         nfolds = 5, standardize = TRUE)
      probs[, k] <- as.numeric(predict(fit_k, X_te, s = "lambda.min", type = "response"))
    }
    pred <- factor(classes[apply(probs, 1, which.max)], levels = classes)
  }
  cm <- caret::confusionMatrix(factor(pred, levels = LABEL_ORDER),
                                factor(y_te, levels = LABEL_ORDER))
  list(cm = cm$table,
       bal_acc = mean(cm$byClass[,"Balanced Accuracy"]),
       midl = cm$table["midl", "midl"],
       n_synth = sum(table(aug$y)) - sum(table(y_tr)))
}

cat("n classes original :", paste(names(table(y_train)), table(y_train),
                                    sep="=", collapse=" | "), "\n")

results <- data.frame()
for (sp in c("raw", "topvar200")) {
  for (fam in c("multinomial", "ovr")) {
    cat(sprintf("\n=== SMOTE %s | %s ===\n", sp, fam))
    set.seed(SEED)
    r <- eval_smote(X_train, y_train, X_test, y_test, sp,
                    target_count = 16, family_type = fam)
    cat(sprintf("  n synthétiques générés : %d\n", r$n_synth))
    cat(sprintf("  Test bal_acc=%.3f | midl=%d/3\n", r$bal_acc, r$midl))
    print(r$cm)
    results <- rbind(results, data.frame(
      smote_space = sp, family = fam,
      test_bal_acc = r$bal_acc,
      midl_correct = r$midl,
      n_synth = r$n_synth))
  }
}

cat("\n========== SYNTHESE NB22 ==========\n")
print(results, row.names=FALSE)

saveRDS(list(results=results),
        "../synthesis/nb22_smote_results.rds")
cat("\n✓ NB22 terminé\n")
