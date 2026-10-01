# =====================================================================
# NB21 — SGCCA avec pondération des classes en amont
# =====================================================================
# Patron suivi : NB09 (rgcca avec blocks$y FACTOR + response=3).
# Pondération : on multiplie chaque ligne i de chaque bloc explicatif
# par sqrt(w_i / mean(w)), ce qui revient à maximiser la covariance
# pondérée dans le critère SGCCA.
# =====================================================================

setwd("/Users/ruben/Documents/Brain-Cancer-Prediction-Model/models")
suppressPackageStartupMessages({
  library(RGCCA); library(MASS); library(glmnet)
  library(data.table); library(caret)
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

# ---- Poids inv. prévalence ----
n_tr <- length(y_train); K <- 3
n_c <- table(y_train)
w_class <- n_tr / (K * n_c)
w <- unname(w_class[as.character(y_train)])
sqrt_w <- sqrt(w / mean(w))
cat("Poids par classe :", round(w_class, 3),
    "| sqrt_w sample :", round(range(sqrt_w), 3), "\n")

# ---- Aval : LDA (avec prior éventuel) ou glmnet ----
predict_with_lda <- function(comp_tr, y_tr, comp_te, priors = NULL) {
  if (is.null(priors)) fit <- lda(comp_tr, grouping = y_tr)
  else                  fit <- lda(comp_tr, grouping = y_tr, prior = priors)
  predict(fit, comp_te)$class
}
predict_with_glmnet <- function(comp_tr, y_tr, comp_te, weights = NULL) {
  if (is.null(weights)) weights <- rep(1, length(y_tr))
  fit <- cv.glmnet(comp_tr, y_tr, family = "multinomial",
                   type.multinomial = "ungrouped", alpha = 0,
                   nfolds = 5, weights = weights, standardize = TRUE)
  predict(fit, comp_te, s = "lambda.min", type = "class")[, 1]
}

# ---- Fit SGCCA puis classifieur aval ----
fit_pipeline <- function(blocks_train, blocks_test, y_train, y_test,
                          downstream = c("lda_empirique", "lda_uniforme",
                                          "glmnet_pondere"),
                          weights_glmnet = NULL) {
  fit_sg <- rgcca(blocks = blocks_train,
                  response = 3,
                  sparsity = c(0.2, 0.2, 1),    # plus généreux pour éviter collapse
                  ncomp = c(1, 1, 1),
                  scheme = "factorial",
                  scale = TRUE, scale_block = TRUE,
                  method = "sgcca",
                  verbose = FALSE)
  # Composantes train
  comp_tr <- cbind(GE = fit_sg$Y[[1]][,1], CGH = fit_sg$Y[[2]][,1])
  # Projection des blocs test
  proj_te <- rgcca_transform(fit_sg, blocks_test = blocks_test)
  comp_te <- cbind(GE = proj_te[[1]][,1], CGH = proj_te[[2]][,1])

  results <- list()
  for (mode in downstream) {
    pred <- switch(mode,
      "lda_empirique"  = predict_with_lda(comp_tr, y_train, comp_te, NULL),
      "lda_uniforme"   = predict_with_lda(comp_tr, y_train, comp_te, rep(1/3, 3)),
      "glmnet_pondere" = predict_with_glmnet(comp_tr, y_train, comp_te, weights_glmnet))
    cm <- caret::confusionMatrix(factor(pred, levels = LABEL_ORDER),
                                  factor(y_test, levels = LABEL_ORDER))
    results[[mode]] <- list(
      bal_acc = mean(cm$byClass[,"Balanced Accuracy"]),
      acc = unname(cm$overall["Accuracy"]),
      midl_correct = cm$table["midl","midl"],
      cm = cm$table)
  }
  list(results = results,
       n_ge  = sum(abs(fit_sg$a[[1]]) > 1e-8),
       n_cgh = sum(abs(fit_sg$a[[2]]) > 1e-8))
}

# ---- Blocs : standard vs pondéré ----
blocks_std_tr <- list(GE = GE_tr, CGH = CGH_tr, y = y_train)
blocks_std_te <- list(GE = GE_te, CGH = CGH_te)

GE_w_tr  <- sweep(GE_tr,  1, sqrt_w, "*")
CGH_w_tr <- sweep(CGH_tr, 1, sqrt_w, "*")
blocks_w_tr <- list(GE = GE_w_tr, CGH = CGH_w_tr, y = y_train)

cat("\n=== SGCCA STANDARD (baseline) ===\n")
out_std <- fit_pipeline(blocks_std_tr, blocks_std_te, y_train, y_test,
                         weights_glmnet = w)
for (m in names(out_std$results)) {
  r <- out_std$results[[m]]
  cat(sprintf("  %-20s : bal_acc=%.3f | midl=%d/3\n",
              m, r$bal_acc, r$midl_correct))
}
cat(sprintf("  Variables : %d GE | %d CGH\n", out_std$n_ge, out_std$n_cgh))

cat("\n=== SGCCA PONDEREE (sqrt_w upstream) ===\n")
out_w <- fit_pipeline(blocks_w_tr, blocks_std_te, y_train, y_test,
                       weights_glmnet = w)
for (m in names(out_w$results)) {
  r <- out_w$results[[m]]
  cat(sprintf("  %-20s : bal_acc=%.3f | midl=%d/3\n",
              m, r$bal_acc, r$midl_correct))
}
cat(sprintf("  Variables : %d GE | %d CGH\n", out_w$n_ge, out_w$n_cgh))

# ---- Tableau de synthèse ----
synth <- data.frame()
for (sg_kind in c("std", "weighted")) {
  out <- if (sg_kind == "std") out_std else out_w
  for (m in names(out$results)) {
    r <- out$results[[m]]
    synth <- rbind(synth, data.frame(
      sgcca = sg_kind, downstream = m,
      test_bal_acc = r$bal_acc,
      midl_correct = r$midl_correct,
      n_ge = out$n_ge, n_cgh = out$n_cgh))
  }
}
cat("\n========== SYNTHESE NB21 ==========\n")
print(synth, row.names=FALSE)

saveRDS(list(synth=synth,
              out_std=out_std, out_w=out_w,
              weights=w, w_class=w_class),
        "../synthesis/nb21_sgcca_weights_results.rds")
cat("\n✓ NB21 terminé\n")
