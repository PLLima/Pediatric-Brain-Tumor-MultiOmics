# =====================================================================
# NB20 — Fused Lasso (Total Variation) sur le bloc CGH
# =====================================================================
# Pénalité demandée par Tenenhaus :
#   lambda_TV * sum_{j>=2} |beta_CGH[j] - beta_CGH[j-1]|
# Encourage des coefficients constants par morceaux le long des
# segments CGH adjacents (=> regroupement spontané en "régions").
#
# Implémentation : reparamétrisation par sommes cumulatives.
# Soit gamma[1] = beta[1] et gamma[j] = beta[j] - beta[j-1] pour j >= 2.
# Alors beta = L %*% gamma (L lower-triangular d'uns) et
#   sum_{j>=2} |gamma[j]| = ||beta||_TV
# La transformation Z_tilde = Z %*% L revient à remplacer Z par sa
# somme cumulative à droite : (Z_tilde)_{i,j} = sum_{k >= j} Z_{i,k}.
# Le L1 standard de glmnet sur gamma[2:p] = TV sur beta.
#
# On garde L1 sur le bloc GE (Lasso classique).
# OvR binomial pour gérer midl (cf. NB11).
# =====================================================================

setwd("/Users/ruben/Documents/Brain-Cancer-Prediction-Model/models")
suppressPackageStartupMessages({
  library(glmnet); library(data.table); library(caret)
})
SEED <- 42; set.seed(SEED)
LABEL_ORDER <- c("cort", "dipg", "midl")
data_dir <- "../data"

# ---- Helpers ----
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

# ---- Tri CGH par ID numérique (proxy adjacence génomique) ----
cgh_ids <- as.numeric(colnames(CGH_tr))
ord <- order(cgh_ids)
CGH_tr_s <- CGH_tr[, ord]; CGH_te_s <- CGH_te[, ord]
p_CGH <- ncol(CGH_tr_s); p_GE <- ncol(GE_tr)

# ---- Reparamétrisation : Z_tilde = Z %*% L (lower triangular) ----
# (Z_tilde)_{i,j} = sum_{k >= j} Z_{i,k}  =  right-cumsum
right_cumsum <- function(X) t(apply(X, 1, function(r) rev(cumsum(rev(r)))))
CGH_tr_tilde <- right_cumsum(CGH_tr_s)
CGH_te_tilde <- right_cumsum(CGH_te_s)
cat(sprintf("Reparam OK : CGH %d × %d -> %d × %d (idem dim, sommes cumulatives droite)\n",
            nrow(CGH_tr_s), ncol(CGH_tr_s), nrow(CGH_tr_tilde), ncol(CGH_tr_tilde)))

X_train <- cbind(GE_tr, CGH_tr_tilde); X_test <- cbind(GE_te, CGH_te_tilde)
colnames(X_train)[1:p_GE] <- paste0("GE__", colnames(GE_tr))
colnames(X_train)[(p_GE+1):(p_GE+p_CGH)] <- paste0("gCGH__", colnames(CGH_tr_s))
colnames(X_test) <- colnames(X_train)

# ---- penalty.factor : 1 pour GE (L1), 0 pour gamma_CGH[1], 1 pour gamma_CGH[2:p] ----
pf <- c(rep(1, p_GE), 0, rep(1, p_CGH - 1))
cat(sprintf("Penalty factor : GE = 1 (Lasso), CGH = [0, 1, 1, ..., 1] (TV pure)\n"))

# =====================================================================
# OvR Lasso + TV (CV 7×3)
# =====================================================================
set.seed(SEED)
outer_folds <- caret::createMultiFolds(y_train, k=7, times=3)

cv_one_fold <- function(tr_idx, va_idx) {
  classes <- LABEL_ORDER
  probs <- matrix(0, length(va_idx), length(classes)); colnames(probs) <- classes
  for (k in seq_along(classes)) {
    y_k <- as.numeric(y_train[tr_idx] == classes[k])
    fit_k <- tryCatch(
      cv.glmnet(X_train[tr_idx,], y_k, family="binomial", alpha=1,
                 nfolds=5, standardize=TRUE, penalty.factor=pf),
      error=function(e) NULL)
    if (is.null(fit_k)) { probs[,k] <- NA; next }
    probs[,k] <- as.numeric(predict(fit_k, X_train[va_idx,],
                                      s="lambda.min", type="response"))
  }
  probs
}

cat("\n=== CV NB20 — OvR Lasso + TV sur CGH ===\n")
scores <- numeric(length(outer_folds))
midl_recalls <- numeric(length(outer_folds))
t0 <- Sys.time()
for (i in seq_along(outer_folds)) {
  tr_idx <- outer_folds[[i]]
  va_idx <- setdiff(seq_along(y_train), tr_idx)
  probs <- cv_one_fold(tr_idx, va_idx)
  if (any(is.na(probs))) { scores[i] <- NA; midl_recalls[i] <- NA; next }
  pred <- factor(LABEL_ORDER[apply(probs, 1, which.max)], levels=LABEL_ORDER)
  cm <- caret::confusionMatrix(pred, factor(y_train[va_idx], levels=LABEL_ORDER))
  scores[i] <- mean(cm$byClass[,"Balanced Accuracy"], na.rm=TRUE)
  midl_recalls[i] <- cm$byClass["Class: midl","Sensitivity"]
  if (i %% 7 == 0)
    cat(sprintf("  fold %d/%d (%.1f min)\n", i, length(outer_folds),
                as.numeric(difftime(Sys.time(),t0,units="mins"))))
}
cv_tv <- list(
  mean_bal_acc=mean(scores, na.rm=TRUE),
  sd_bal_acc=sd(scores, na.rm=TRUE),
  mean_midl=mean(midl_recalls, na.rm=TRUE),
  sd_midl=sd(midl_recalls, na.rm=TRUE),
  n_folds=sum(!is.na(scores)))
cat(sprintf("\nTV OvR CV : %.3f ± %.3f | midl recall %.3f ± %.3f\n",
            cv_tv$mean_bal_acc, cv_tv$sd_bal_acc,
            cv_tv$mean_midl, cv_tv$sd_midl))

# =====================================================================
# Refit final + test + récupération beta_CGH (= cumsum gauche de gamma)
# =====================================================================
classes <- LABEL_ORDER
probs_test <- matrix(0, nrow(X_test), length(classes)); colnames(probs_test) <- classes
beta_cgh_per_class <- list()
for (k in seq_along(classes)) {
  y_k <- as.numeric(y_train == classes[k])
  fit_k <- cv.glmnet(X_train, y_k, family="binomial", alpha=1,
                      nfolds=10, standardize=TRUE, penalty.factor=pf)
  probs_test[,k] <- as.numeric(predict(fit_k, X_test, s="lambda.min", type="response"))
  cf <- coef(fit_k, s="lambda.min")[-1, 1]
  # gamma_CGH = cf[(p_GE+1):(p_GE+p_CGH)]
  gamma_cgh <- cf[(p_GE+1):(p_GE+p_CGH)]
  beta_cgh  <- cumsum(gamma_cgh)   # gamma -> beta via cumsum gauche
  beta_cgh_per_class[[classes[k]]] <- list(
    gamma = gamma_cgh, beta = beta_cgh,
    n_breakpoints = sum(abs(gamma_cgh[-1]) > 1e-8),    # nb de "sauts" TV
    n_active_segs = sum(abs(beta_cgh) > 1e-8),         # nb segments à coef != 0
    n_active_ge = sum(abs(cf[1:p_GE]) > 1e-8))
  cat(sprintf("Classe %s : %d GE | %d CGH actifs | %d sauts TV\n",
              classes[k], beta_cgh_per_class[[classes[k]]]$n_active_ge,
              beta_cgh_per_class[[classes[k]]]$n_active_segs,
              beta_cgh_per_class[[classes[k]]]$n_breakpoints))
}
pred_test <- factor(classes[apply(probs_test, 1, which.max)], levels=classes)
cm_test <- caret::confusionMatrix(pred_test, factor(y_test, levels=LABEL_ORDER))
cat("\n=== TEST NB20 ===\n")
print(cm_test$table)
cat(sprintf("Acc %.3f | Bal_acc %.3f\n",
            cm_test$overall["Accuracy"],
            mean(cm_test$byClass[,"Balanced Accuracy"])))

saveRDS(list(cv=cv_tv, cm_test=cm_test$table,
             test_bal_acc=mean(cm_test$byClass[,"Balanced Accuracy"]),
             midl_test_correct=cm_test$table["midl","midl"],
             beta_cgh_per_class=beta_cgh_per_class,
             cgh_order=cgh_ids[ord]),
        "../synthesis/nb20_fused_lasso_results.rds")
cat("\n✓ NB20 terminé\n")
