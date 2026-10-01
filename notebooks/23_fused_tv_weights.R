# =====================================================================
# NB23 — Fused Lasso TV sur CGH + pondération des classes
# =====================================================================
# Combine deux ingrédients déjà testés isolément :
#   - NB20 : reparamétrisation par sommes cumulatives droites pour TV
#   - NB17 : pondération inv_prevalence dans cv.glmnet
#
# Objectif : voir si les deux mécanismes se cumulent — TV pour
# respecter la structure spatiale CGH + weights pour récupérer midl.
# =====================================================================

setwd("/Users/ruben/Documents/Brain-Cancer-Prediction-Model/models")
suppressPackageStartupMessages({
  library(glmnet); library(data.table); library(caret)
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

cgh_ids <- as.numeric(colnames(CGH_tr)); ord <- order(cgh_ids)
CGH_tr_s <- CGH_tr[, ord]; CGH_te_s <- CGH_te[, ord]
p_CGH <- ncol(CGH_tr_s); p_GE <- ncol(GE_tr)
right_cumsum <- function(X) t(apply(X, 1, function(r) rev(cumsum(rev(r)))))
CGH_tr_tilde <- right_cumsum(CGH_tr_s)
CGH_te_tilde <- right_cumsum(CGH_te_s)
X_train <- cbind(GE_tr, CGH_tr_tilde); X_test <- cbind(GE_te, CGH_te_tilde)
pf <- c(rep(1, p_GE), 0, rep(1, p_CGH - 1))

# ---- Poids inv. prévalence ----
n_tr <- length(y_train); K <- 3
n_c <- table(y_train)
w_class <- n_tr / (K * n_c)
w <- unname(w_class[as.character(y_train)])
cat("Poids inv. prévalence par classe :", round(w_class, 3), "\n")

# ---- Fit OvR Lasso + TV avec/sans poids ----
fit_tv_ovr <- function(X_tr, y_tr, X_te, weights = NULL) {
  if (is.null(weights)) weights <- rep(1, length(y_tr))
  classes <- LABEL_ORDER
  probs <- matrix(0, nrow(X_te), length(classes)); colnames(probs) <- classes
  beta_per_class <- list()
  for (k in seq_along(classes)) {
    y_k <- as.numeric(y_tr == classes[k])
    fit_k <- cv.glmnet(X_tr, y_k, family = "binomial", alpha = 1,
                       nfolds = 10, standardize = TRUE,
                       penalty.factor = pf, weights = weights)
    probs[, k] <- as.numeric(predict(fit_k, X_te, s = "lambda.min", type = "response"))
    cf <- coef(fit_k, s = "lambda.min")[-1, 1]
    gamma_cgh <- cf[(p_GE + 1):length(cf)]
    beta_cgh  <- cumsum(gamma_cgh)
    beta_per_class[[classes[k]]] <- list(
      gamma = gamma_cgh, beta = beta_cgh,
      n_breakpoints = sum(abs(gamma_cgh[-1]) > 1e-8),
      n_active_segs = sum(abs(beta_cgh) > 1e-8),
      n_active_ge = sum(abs(cf[1:p_GE]) > 1e-8))
  }
  pred <- factor(classes[apply(probs, 1, which.max)], levels = classes)
  list(probs = probs, pred = pred, beta_per_class = beta_per_class)
}

# ---- Fit multinomial + TV avec/sans poids ----
fit_tv_multi <- function(X_tr, y_tr, X_te, weights = NULL) {
  if (is.null(weights)) weights <- rep(1, length(y_tr))
  fit <- cv.glmnet(X_tr, y_tr, family = "multinomial",
                   type.multinomial = "ungrouped", alpha = 1,
                   nfolds = 10, standardize = TRUE,
                   penalty.factor = pf, weights = weights)
  pred <- predict(fit, X_te, s = "lambda.min", type = "class")[, 1]
  cf_list <- coef(fit, s = "lambda.min")
  beta_per_class <- list()
  for (cl in names(cf_list)) {
    cf <- cf_list[[cl]][-1, 1]
    gamma_cgh <- cf[(p_GE + 1):length(cf)]
    beta_per_class[[cl]] <- list(
      n_breakpoints = sum(abs(gamma_cgh[-1]) > 1e-8),
      n_active_ge = sum(abs(cf[1:p_GE]) > 1e-8))
  }
  list(pred = pred, beta_per_class = beta_per_class)
}

# ---- 4 variantes ----
variantes <- list(
  list(name="TV OvR sans poids",       fn=fit_tv_ovr,   w=NULL),
  list(name="TV OvR + weights inv_prev", fn=fit_tv_ovr, w=w),
  list(name="TV Multi sans poids",     fn=fit_tv_multi, w=NULL),
  list(name="TV Multi + weights inv_prev", fn=fit_tv_multi, w=w)
)

synth <- data.frame()
for (v in variantes) {
  set.seed(SEED)
  res <- v$fn(X_train, y_train, X_test, weights = v$w)
  cm <- caret::confusionMatrix(factor(res$pred, levels = LABEL_ORDER),
                                factor(y_test, levels = LABEL_ORDER))
  ba <- mean(cm$byClass[,"Balanced Accuracy"])
  midl <- cm$table["midl", "midl"]
  cat(sprintf("\n=== %s ===\n", v$name))
  print(cm$table)
  cat(sprintf("Bal_acc=%.3f | midl=%d/3\n", ba, midl))
  # Breakpoints moyens
  bps <- sapply(res$beta_per_class, function(x) x$n_breakpoints)
  ges <- sapply(res$beta_per_class, function(x) x$n_active_ge)
  synth <- rbind(synth, data.frame(
    variante = v$name,
    test_bal_acc = ba,
    midl_correct = midl,
    avg_breakpoints_TV = mean(bps),
    avg_n_active_GE = mean(ges)))
}

cat("\n========== SYNTHESE NB23 ==========\n")
print(synth, row.names=FALSE)

saveRDS(list(synth=synth, weights=w_class),
        "../synthesis/nb23_tv_weights_results.rds")
cat("\n✓ NB23 terminé\n")
