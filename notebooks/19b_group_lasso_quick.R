# =====================================================================
# NB19b — Group Lasso CGH (version rapide : test + analyse uniquement)
# =====================================================================
# Skip la CV qui est trop coûteuse (les CV étaient en cours mais
# memory-bound). On garde uniquement le refit final + analyse des
# groupes CGH actifs, ce qui est suffisant pour le rapport.
# =====================================================================

setwd("/Users/ruben/Documents/Brain-Cancer-Prediction-Model/models")
suppressPackageStartupMessages({
  library(gglasso); library(data.table); library(caret)
})
SEED <- 42; set.seed(SEED)
LABEL_ORDER <- c("cort", "dipg", "midl")
data_dir <- "../data"
SEGMENTS_PER_GROUP <- 50

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

cgh_ids <- as.numeric(colnames(CGH_tr)); ord <- order(cgh_ids)
CGH_tr_s <- CGH_tr[, ord]; CGH_te_s <- CGH_te[, ord]
p_CGH <- ncol(CGH_tr_s); p_GE <- ncol(GE_tr)
n_groups <- ceiling(p_CGH / SEGMENTS_PER_GROUP)
groups_cgh <- rep(1:n_groups, each=SEGMENTS_PER_GROUP)[1:p_CGH]
X_train <- cbind(GE_tr, CGH_tr_s); X_test <- cbind(GE_te, CGH_te_s)
groups_int <- as.integer(as.factor(c(seq_len(p_GE), p_GE + groups_cgh)))
cat(sprintf("CGH : %d segments, %d pseudo-groupes (~%d segs)\n",
            p_CGH, n_groups, SEGMENTS_PER_GROUP))

# ---- Refit final OvR (3 classes) ----
classes <- LABEL_ORDER
probs_test <- matrix(0, nrow(X_test), length(classes)); colnames(probs_test) <- classes
active_per_class <- list()
for (k in seq_along(classes)) {
  cat(sprintf("Fit classe %s...\n", classes[k]))
  y_k <- ifelse(y_train == classes[k], 1, -1)
  t0 <- Sys.time()
  fit_k <- cv.gglasso(X_train, y_k, group=groups_int, loss="logit",
                      nfolds=5, pred.loss="loss")
  cat(sprintf("  cv.gglasso %.1f min\n",
              as.numeric(difftime(Sys.time(),t0,units="mins"))))
  eta <- predict(fit_k$gglasso.fit, newx=X_test, s=fit_k$lambda.min, type="link")
  probs_test[, k] <- 1 / (1 + exp(-eta))
  beta <- coef(fit_k$gglasso.fit, s=fit_k$lambda.min)[-1, 1]
  cgh_beta <- beta[(p_GE+1):length(beta)]
  active_cgh_segs <- which(abs(cgh_beta) > 1e-8)
  active_groups <- unique(groups_cgh[active_cgh_segs])
  active_ge <- sum(abs(beta[1:p_GE]) > 1e-8)
  active_per_class[[classes[k]]] <- list(
    n_ge=active_ge, n_cgh_segs=length(active_cgh_segs),
    n_cgh_groups=length(active_groups), groups=active_groups)
  cat(sprintf("  %s : %d GE | %d segs CGH | %d/%d groupes CGH actifs\n",
              classes[k], active_ge, length(active_cgh_segs),
              length(active_groups), n_groups))
  gc()
}
pred_test <- factor(classes[apply(probs_test, 1, which.max)], levels=classes)
cm_test <- caret::confusionMatrix(pred_test, factor(y_test, levels=LABEL_ORDER))
cat("\n=== TEST NB19 ===\n")
print(cm_test$table)
cat(sprintf("Acc %.3f | Bal_acc %.3f\n",
            cm_test$overall["Accuracy"],
            mean(cm_test$byClass[,"Balanced Accuracy"])))

saveRDS(list(cm_test=cm_test$table,
             test_bal_acc=mean(cm_test$byClass[,"Balanced Accuracy"]),
             test_acc=cm_test$overall["Accuracy"],
             midl_test_correct=cm_test$table["midl","midl"],
             active_per_class=active_per_class,
             n_groups=n_groups, segs_per_group=SEGMENTS_PER_GROUP),
        "../synthesis/nb19_group_lasso_results.rds")
cat("\n✓ NB19b terminé\n")
