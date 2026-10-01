# =====================================================================
# NB19 — Group Lasso par groupes pseudo-chromosomiques sur CGH
# =====================================================================
# Les IDs de segments CGH (1, 3, 5, 6, ..., 1229) sont numériques.
# Sans annotation chromosomique disponible, on construit des
# pseudo-groupes par tranches consécutives de SEGMENTS_PER_GROUP IDs.
# Cela approche la notion de "région chromosomique" et permet de tester
# l'hypothèse de Tenenhaus : la sélection devrait respecter la proximité
# numérique (=> spatiale) des segments.
#
# Pénalité Group Lasso :
#   sum_g w_g ||beta_g||_2     (Yuan & Lin 2006)
# Un groupe entier est gardé ou retiré.
#
# Package R : gglasso (Group Lasso) — binomial OvR puis argmax.
# =====================================================================

setwd("/Users/ruben/Documents/Brain-Cancer-Prediction-Model/models")
suppressPackageStartupMessages({
  if (!requireNamespace("gglasso", quietly=TRUE))
    install.packages("gglasso", repos="https://cloud.r-project.org")
  library(gglasso); library(glmnet); library(data.table); library(caret)
})

SEED <- 42; set.seed(SEED)
LABEL_ORDER <- c("cort", "dipg", "midl")
data_dir <- "../data"
SEGMENTS_PER_GROUP <- 50   # ~25 pseudo-chromosomes sur 1229 segments

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

# ---- Construction des pseudo-groupes ----
# Tri par ID numérique pour préserver l'adjacence
cgh_ids <- as.numeric(colnames(CGH_tr))
ord <- order(cgh_ids)
CGH_tr_s <- CGH_tr[, ord]; CGH_te_s <- CGH_te[, ord]
p_cgh <- ncol(CGH_tr_s)
n_groups <- ceiling(p_cgh / SEGMENTS_PER_GROUP)
groups_cgh <- rep(1:n_groups, each=SEGMENTS_PER_GROUP)[1:p_cgh]
cat(sprintf("CGH : %d segments → %d pseudo-groupes de ~%d segments\n",
            p_cgh, n_groups, SEGMENTS_PER_GROUP))

# Concaténation X = [GE | CGH] avec groupes :
# GE : chaque gène = singleton ; CGH : pseudo-groupes
X_train <- cbind(GE_tr, CGH_tr_s); X_test <- cbind(GE_te, CGH_te_s)
p_GE <- ncol(GE_tr)
groups_int <- c(seq_len(p_GE),
                p_GE + as.integer(groups_cgh))
# gglasso exige des groupes consécutifs commençant à 1
groups_int <- as.integer(as.factor(groups_int))
cat(sprintf("Total groupes : %d (GE singletons : %d, CGH groupes : %d)\n",
            length(unique(groups_int)), p_GE, n_groups))

# =====================================================================
# CV OvR avec Group Lasso (gglasso family="logit")
# =====================================================================
set.seed(SEED)
outer_folds <- caret::createMultiFolds(y_train, k=7, times=3)

fit_gglasso_ovr <- function(X_tr, y_tr, X_va, groups_int) {
  classes <- LABEL_ORDER
  probs <- matrix(0, nrow(X_va), length(classes)); colnames(probs) <- classes
  for (k in seq_along(classes)) {
    y_k <- ifelse(y_tr == classes[k], 1, -1)   # gglasso veut {-1,+1}
    fit <- tryCatch(
      cv.gglasso(X_tr, y_k, group=groups_int, loss="logit",
                  nfolds=5, pred.loss="loss"),
      error=function(e) NULL)
    if (is.null(fit)) { probs[,k] <- NA; next }
    # Logit -> proba sigmoid
    eta <- predict(fit$gglasso.fit, newx=X_va, s=fit$lambda.min, type="link")
    probs[,k] <- 1 / (1 + exp(-eta))
  }
  probs
}

cat("\n=== CV NB19 — Group Lasso OvR ===\n")
scores_gl <- numeric(length(outer_folds))
midl_recalls_gl <- numeric(length(outer_folds))
t0 <- Sys.time()
for (i in seq_along(outer_folds)) {
  tr_idx <- outer_folds[[i]]
  va_idx <- setdiff(seq_along(y_train), tr_idx)
  probs <- fit_gglasso_ovr(X_train[tr_idx,], y_train[tr_idx],
                            X_train[va_idx,], groups_int)
  if (any(is.na(probs))) { scores_gl[i] <- NA; midl_recalls_gl[i] <- NA; next }
  pred <- factor(LABEL_ORDER[apply(probs, 1, which.max)], levels=LABEL_ORDER)
  cm <- caret::confusionMatrix(pred, factor(y_train[va_idx], levels=LABEL_ORDER))
  scores_gl[i] <- mean(cm$byClass[,"Balanced Accuracy"], na.rm=TRUE)
  midl_recalls_gl[i] <- cm$byClass["Class: midl","Sensitivity"]
  if (i %% 5 == 0)
    cat(sprintf("  fold %d/%d (%.1f min)\n", i, length(outer_folds),
                as.numeric(difftime(Sys.time(),t0,units="mins"))))
}
cv_glasso <- list(
  mean_bal_acc=mean(scores_gl, na.rm=TRUE),
  sd_bal_acc=sd(scores_gl, na.rm=TRUE),
  mean_midl=mean(midl_recalls_gl, na.rm=TRUE),
  sd_midl=sd(midl_recalls_gl, na.rm=TRUE),
  n_folds=sum(!is.na(scores_gl)))
cat(sprintf("\nGroup Lasso OvR CV : %.3f ± %.3f | midl recall %.3f ± %.3f\n",
            cv_glasso$mean_bal_acc, cv_glasso$sd_bal_acc,
            cv_glasso$mean_midl, cv_glasso$sd_midl))

# =====================================================================
# Refit final + test
# =====================================================================
probs_test <- fit_gglasso_ovr(X_train, y_train, X_test, groups_int)
pred_test <- factor(LABEL_ORDER[apply(probs_test, 1, which.max)], levels=LABEL_ORDER)
cm_test <- caret::confusionMatrix(pred_test, factor(y_test, levels=LABEL_ORDER))
cat("\n=== TEST NB19 ===\n")
print(cm_test$table)
cat(sprintf("Accuracy %.3f | Balanced %.3f\n",
            cm_test$overall["Accuracy"],
            mean(cm_test$byClass[,"Balanced Accuracy"])))

# Combien de pseudo-groupes (chromosomes) CGH sont retenus par classe ?
cat("\nGroupes pseudo-chromosomiques CGH actifs par classe :\n")
groups_cgh_active <- list()
for (k in seq_along(LABEL_ORDER)) {
  y_k <- ifelse(y_train == LABEL_ORDER[k], 1, -1)
  fit_k <- cv.gglasso(X_train, y_k, group=groups_int, loss="logit",
                      nfolds=5, pred.loss="loss")
  beta <- coef(fit_k$gglasso.fit, s=fit_k$lambda.min)[-1, 1]
  cgh_beta <- beta[(p_GE + 1):length(beta)]
  active_cgh_segs <- which(abs(cgh_beta) > 1e-8)
  active_groups   <- unique(groups_cgh[active_cgh_segs])
  active_ge       <- sum(abs(beta[1:p_GE]) > 1e-8)
  cat(sprintf("  %s : %d GE actifs | %d/%d pseudo-groupes CGH actifs\n",
              LABEL_ORDER[k], active_ge, length(active_groups), n_groups))
  groups_cgh_active[[LABEL_ORDER[k]]] <- active_groups
}

saveRDS(list(cv=cv_glasso, cm_test=cm_test$table,
             test_bal_acc=mean(cm_test$byClass[,"Balanced Accuracy"]),
             midl_test_correct=cm_test$table["midl","midl"],
             groups_cgh_active=groups_cgh_active,
             n_groups=n_groups, segs_per_group=SEGMENTS_PER_GROUP),
        "../synthesis/nb19_group_lasso_results.rds")
cat("\n✓ NB19 terminé\n")
