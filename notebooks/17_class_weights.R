# =====================================================================
# NB17 — Gestion du déséquilibre des classes par poids
# =====================================================================
# Reprend NB14c (multinomial ungrouped, alpha=1) et NB11 (OvR binomial)
# en ajoutant des poids w_i = w(Y_i) sur les observations.
#
# Quatre régimes de poids :
#   - "none"           : pas de poids (baseline = NB14c / NB11)
#   - "inv_prevalence" : w_c = n / (K * n_c)
#   - "sqrt_inv"       : w_c = sqrt(n / n_c) -- ponderation douce
#   - "effective_num"  : Cui et al. 2019, w_c = (1-beta)/(1-beta^n_c), beta=0.99
#
# Évalué via CV 7-fold × 3 = 21 plis (mêmes plis que NB14c pour comparaison
# paired Wilcoxon possible).
# =====================================================================

setwd("/Users/ruben/Documents/Brain-Cancer-Prediction-Model/models")
suppressPackageStartupMessages({
  library(glmnet); library(data.table); library(caret)
})

SEED <- 42; set.seed(SEED)
LABEL_ORDER <- c("cort", "dipg", "midl")
data_dir <- "../data"

# ---- Helpers de chargement (identique aux autres NB) ----
to_numeric_frame <- function(df) {
  rn <- rownames(df)
  out <- as.data.frame(
    lapply(df, function(x) as.numeric(gsub(",", ".", as.character(x), fixed=TRUE))),
    check.names=FALSE); rownames(out) <- rn; out
}
load_block <- function(blk, split) {
  df <- as.data.frame(fread(file.path(data_dir,
    sprintf("ge_cgh_locIGR__multiblocks__%s__%s.csv", blk, split)),
    check.names=FALSE))
  rownames(df) <- as.character(df[[1]]); df[[1]] <- NULL
  to_numeric_frame(df)
}
load_y <- function(split) {
  df <- as.data.frame(fread(file.path(data_dir,
    sprintf("ge_cgh_locIGR__multiblocks__y__%s.csv", split)),
    check.names=FALSE))
  ids <- as.character(df[[1]]); df[[1]] <- NULL
  y <- factor(LABEL_ORDER[max.col(as.matrix(df[, LABEL_ORDER]), ties.method="first")],
              levels=LABEL_ORDER)
  names(y) <- ids; y
}

GE_tr  <- load_block("GE",  "train"); GE_te  <- load_block("GE",  "test")
CGH_tr <- load_block("CGH", "train"); CGH_te <- load_block("CGH", "test")
y_tr_raw <- load_y("train"); y_te_raw <- load_y("test")

tr_ids <- Reduce(intersect, list(rownames(GE_tr), rownames(CGH_tr), names(y_tr_raw)))
te_ids <- Reduce(intersect, list(rownames(GE_te), rownames(CGH_te), names(y_te_raw)))

GE_tr  <- as.matrix(GE_tr[tr_ids,]);  CGH_tr <- as.matrix(CGH_tr[tr_ids,])
GE_te  <- as.matrix(GE_te[te_ids,]);  CGH_te <- as.matrix(CGH_te[te_ids,])
y_train <- y_tr_raw[tr_ids]; y_test <- y_te_raw[te_ids]

# Impute median
imp <- function(tr, te) {
  med <- apply(tr, 2, median, na.rm=TRUE)
  for (j in 1:ncol(tr)) {
    tr[is.na(tr[,j]),j] <- med[j]; te[is.na(te[,j]),j] <- med[j]
  }; list(tr=tr, te=te)
}
f <- imp(GE_tr, GE_te);   GE_tr <- f$tr;  GE_te <- f$te
f <- imp(CGH_tr, CGH_te); CGH_tr <- f$tr; CGH_te <- f$te

X_train <- cbind(GE_tr, CGH_tr); X_test <- cbind(GE_te, CGH_te)
colnames(X_train)[1:ncol(GE_tr)] <- paste0("GE__", colnames(GE_tr))
colnames(X_train)[(ncol(GE_tr)+1):ncol(X_train)] <- paste0("CGH__", colnames(CGH_tr))
colnames(X_test) <- colnames(X_train)

cat(sprintf("Train %d × %d | Test %d\n", nrow(X_train), ncol(X_train), nrow(X_test)))
n_tr <- length(y_train); K <- 3
n_per_class <- table(y_train)
cat("Classes train :", paste(names(n_per_class), n_per_class, sep="=", collapse=" | "), "\n")

# =====================================================================
# Schémas de poids
# =====================================================================
make_weights <- function(y, scheme) {
  n <- length(y); n_c <- table(y); K <- length(n_c)
  if (scheme == "none") return(rep(1, n))
  if (scheme == "inv_prevalence") w_c <- n / (K * n_c)
  if (scheme == "sqrt_inv")       w_c <- sqrt(n / n_c)
  if (scheme == "effective_num") {
    beta <- 0.99
    w_c <- (1 - beta) / (1 - beta^as.numeric(n_c))
    w_c <- w_c / sum(w_c) * K       # normalise
  }
  unname(w_c[as.character(y)])
}

# =====================================================================
# CV pipeline générique
# =====================================================================
set.seed(SEED)
outer_folds <- caret::createMultiFolds(y_train, k=7, times=3)

eval_one <- function(scheme, family_type) {
  scores <- numeric(length(outer_folds))
  midl_recalls <- numeric(length(outer_folds))
  for (i in seq_along(outer_folds)) {
    tr_idx <- outer_folds[[i]]
    va_idx <- setdiff(seq_along(y_train), tr_idx)
    w <- make_weights(y_train[tr_idx], scheme)

    if (family_type == "multinomial") {
      fit <- tryCatch(cv.glmnet(X_train[tr_idx,], y_train[tr_idx],
                                 family="multinomial",
                                 type.multinomial="ungrouped",
                                 alpha=1, nfolds=5, weights=w,
                                 standardize=TRUE),
                      error=function(e) NULL)
      if (is.null(fit)) { scores[i] <- NA; midl_recalls[i] <- NA; next }
      pred <- predict(fit, X_train[va_idx,], s="lambda.min", type="class")[,1]
    } else if (family_type == "ovr") {
      classes <- levels(y_train)
      probs <- matrix(0, length(va_idx), length(classes))
      colnames(probs) <- classes
      for (k in seq_along(classes)) {
        y_k <- as.numeric(y_train[tr_idx] == classes[k])
        fit_k <- tryCatch(cv.glmnet(X_train[tr_idx,], y_k,
                                     family="binomial", alpha=1,
                                     nfolds=5, weights=w,
                                     standardize=TRUE),
                          error=function(e) NULL)
        if (is.null(fit_k)) { probs[,k] <- NA; next }
        probs[,k] <- as.numeric(predict(fit_k, X_train[va_idx,],
                                         s="lambda.min", type="response"))
      }
      if (any(is.na(probs))) { scores[i] <- NA; midl_recalls[i] <- NA; next }
      pred <- factor(classes[apply(probs, 1, which.max)], levels=classes)
    }
    cm <- caret::confusionMatrix(factor(pred, levels=LABEL_ORDER),
                                  factor(y_train[va_idx], levels=LABEL_ORDER))
    scores[i]       <- mean(cm$byClass[,"Balanced Accuracy"], na.rm=TRUE)
    midl_recalls[i] <- cm$byClass["Class: midl", "Sensitivity"]
  }
  list(scores=scores, midl=midl_recalls)
}

# =====================================================================
# Sweep weights × family
# =====================================================================
schemes <- c("none", "inv_prevalence", "sqrt_inv", "effective_num")
families <- c("multinomial", "ovr")
results <- data.frame()

for (sch in schemes) {
  for (fam in families) {
    cat(sprintf("\n=== scheme = %s | family = %s ===\n", sch, fam))
    t0 <- Sys.time()
    res <- eval_one(sch, fam)
    el <- difftime(Sys.time(), t0, units="mins")
    results <- rbind(results, data.frame(
      scheme=sch, family=fam,
      mean_bal_acc=mean(res$scores, na.rm=TRUE),
      sd_bal_acc=sd(res$scores, na.rm=TRUE),
      mean_midl_recall=mean(res$midl, na.rm=TRUE),
      sd_midl_recall=sd(res$midl, na.rm=TRUE),
      n_folds=sum(!is.na(res$scores))))
    cat(sprintf("  bal_acc=%.3f±%.3f | midl_recall=%.3f±%.3f (%.1f min)\n",
                tail(results$mean_bal_acc,1), tail(results$sd_bal_acc,1),
                tail(results$mean_midl_recall,1), tail(results$sd_midl_recall,1),
                as.numeric(el)))
  }
}

cat("\n========== RESULTATS NB17 ==========\n")
print(results)

# =====================================================================
# Test final (refit + matrice de confusion par scheme × family)
# =====================================================================
test_results <- data.frame()
cms <- list()
for (sch in schemes) {
  for (fam in families) {
    w_full <- make_weights(y_train, sch)
    if (fam == "multinomial") {
      fit <- cv.glmnet(X_train, y_train, family="multinomial",
                       type.multinomial="ungrouped", alpha=1, nfolds=10,
                       weights=w_full, standardize=TRUE)
      pred <- predict(fit, X_test, s="lambda.min", type="class")[,1]
    } else {
      classes <- levels(y_train)
      probs <- matrix(0, nrow(X_test), length(classes)); colnames(probs) <- classes
      for (k in seq_along(classes)) {
        y_k <- as.numeric(y_train == classes[k])
        fit_k <- cv.glmnet(X_train, y_k, family="binomial", alpha=1,
                            nfolds=10, weights=w_full, standardize=TRUE)
        probs[,k] <- as.numeric(predict(fit_k, X_test, s="lambda.min", type="response"))
      }
      pred <- factor(classes[apply(probs, 1, which.max)], levels=classes)
    }
    cm <- caret::confusionMatrix(factor(pred, levels=LABEL_ORDER),
                                  factor(y_test, levels=LABEL_ORDER))
    cms[[paste(sch, fam, sep="_")]] <- cm$table
    test_results <- rbind(test_results, data.frame(
      scheme=sch, family=fam,
      test_bal_acc=mean(cm$byClass[,"Balanced Accuracy"]),
      test_acc=cm$overall["Accuracy"],
      midl_correct=cm$table["midl","midl"]))
  }
}
cat("\n========== TEST NB17 ==========\n")
print(test_results)

# Sauvegarde
saveRDS(list(cv=results, test=test_results, cms=cms),
        "../synthesis/nb17_class_weights_results.rds")
cat("\n✓ Sauvegarde nb17_class_weights_results.rds\n")
