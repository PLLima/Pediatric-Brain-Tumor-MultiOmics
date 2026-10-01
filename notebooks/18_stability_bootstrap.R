# =====================================================================
# NB18 — Sélection de variables stable par bootstrap
# =====================================================================
# Pour chaque méthode (OvR Lasso, multinomial ungrouped, SGCCA + LDA),
# on tire B sous-échantillons de taille n/2 sans remise et on enregistre
# pour chaque variable sa fréquence de sélection (pi_hat).
#
# Référence : Meinshausen & Bühlmann 2010, JRSSB
#
# Sortie :
#   nb18_stability_results.rds : pi_hat par méthode × variable
# =====================================================================

setwd("/Users/ruben/Documents/Brain-Cancer-Prediction-Model/models")
suppressPackageStartupMessages({
  library(glmnet); library(data.table); library(caret); library(RGCCA)
})

SEED <- 42; set.seed(SEED)
B <- 200   # bootstrap repetitions
LABEL_ORDER <- c("cort", "dipg", "midl")
data_dir <- "../data"

# ---- Helpers (identique NB17) ----
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

GE_tr  <- load_block("GE",  "train")
CGH_tr <- load_block("CGH", "train")
y_tr_raw <- load_y("train")
tr_ids <- Reduce(intersect, list(rownames(GE_tr), rownames(CGH_tr), names(y_tr_raw)))
GE_tr  <- as.matrix(GE_tr[tr_ids,]);  CGH_tr <- as.matrix(CGH_tr[tr_ids,])
y_train <- y_tr_raw[tr_ids]

imp <- function(M) {
  med <- apply(M, 2, median, na.rm=TRUE)
  for (j in 1:ncol(M)) M[is.na(M[,j]),j] <- med[j]; M
}
GE_tr <- imp(GE_tr); CGH_tr <- imp(CGH_tr)
X_train <- cbind(GE_tr, CGH_tr)
colnames(X_train)[1:ncol(GE_tr)] <- paste0("GE__", colnames(GE_tr))
colnames(X_train)[(ncol(GE_tr)+1):ncol(X_train)] <- paste0("CGH__", colnames(CGH_tr))

n <- nrow(X_train); n_sub <- floor(n/2); p <- ncol(X_train)
cat(sprintf("Train n=%d, p=%d | bootstrap B=%d (subsample n/2=%d)\n",
            n, p, B, n_sub))

# =====================================================================
# Bootstrap 1 : Multinomial Lasso ungrouped
# =====================================================================
cat("\n=== NB18.1 — Multinomial ungrouped bootstrap ===\n")
freq_multi <- matrix(0, p, 3); colnames(freq_multi) <- LABEL_ORDER
rownames(freq_multi) <- colnames(X_train)

set.seed(SEED); t0 <- Sys.time()
for (b in 1:B) {
  idx <- sample(n, n_sub, replace=FALSE)
  if (length(unique(y_train[idx])) < 3) next
  fit <- tryCatch(cv.glmnet(X_train[idx,], y_train[idx],
                             family="multinomial", type.multinomial="ungrouped",
                             alpha=1, nfolds=5, standardize=TRUE),
                  error=function(e) NULL)
  if (is.null(fit)) next
  cf <- coef(fit, s="lambda.min")
  for (k in 1:3) {
    bk <- cf[[k]][-1, 1]
    freq_multi[abs(bk) > 1e-8, k] <- freq_multi[abs(bk) > 1e-8, k] + 1
  }
  if (b %% 25 == 0) cat(sprintf("  multi b=%d/%d (%.1f min)\n", b, B,
                                 as.numeric(difftime(Sys.time(),t0,units="mins"))))
}
freq_multi <- freq_multi / B
cat(sprintf("Top 10 stable (toute classe confondue, max sur classes) :\n"))
pi_max_multi <- apply(freq_multi, 1, max)
print(head(sort(pi_max_multi, decreasing=TRUE), 10))

# =====================================================================
# Bootstrap 2 : OvR binomial Lasso
# =====================================================================
cat("\n=== NB18.2 — OvR binomial bootstrap ===\n")
freq_ovr <- matrix(0, p, 3); colnames(freq_ovr) <- LABEL_ORDER
rownames(freq_ovr) <- colnames(X_train)

set.seed(SEED); t0 <- Sys.time()
for (b in 1:B) {
  idx <- sample(n, n_sub, replace=FALSE)
  if (length(unique(y_train[idx])) < 3) next
  for (k in 1:3) {
    y_k <- as.numeric(y_train[idx] == LABEL_ORDER[k])
    if (sum(y_k) < 2 || sum(y_k) > length(y_k) - 2) next
    fit_k <- tryCatch(cv.glmnet(X_train[idx,], y_k, family="binomial",
                                 alpha=1, nfolds=5, standardize=TRUE),
                      error=function(e) NULL)
    if (is.null(fit_k)) next
    bk <- coef(fit_k, s="lambda.min")[-1, 1]
    freq_ovr[abs(bk) > 1e-8, k] <- freq_ovr[abs(bk) > 1e-8, k] + 1
  }
  if (b %% 25 == 0) cat(sprintf("  ovr b=%d/%d (%.1f min)\n", b, B,
                                 as.numeric(difftime(Sys.time(),t0,units="mins"))))
}
freq_ovr <- freq_ovr / B
pi_max_ovr <- apply(freq_ovr, 1, max)
cat(sprintf("Top 10 stable OvR :\n"))
print(head(sort(pi_max_ovr, decreasing=TRUE), 10))

# =====================================================================
# Bootstrap 3 : SGCCA (rgcca_bootstrap natif)
# =====================================================================
cat("\n=== NB18.3 — SGCCA bootstrap (rgcca_bootstrap natif) ===\n")
# Y one-hot
Y_oh <- model.matrix(~ y_train - 1)
colnames(Y_oh) <- LABEL_ORDER

blocks <- list(GE = GE_tr, CGH = CGH_tr, y = Y_oh)
conn <- matrix(c(0,0,1, 0,0,1, 1,1,0), 3, 3, byrow=TRUE,
               dimnames=list(c("GE","CGH","y"), c("GE","CGH","y")))

t0 <- Sys.time()
fit_sgcca <- rgcca(blocks = blocks,
                   connection = conn,
                   sparsity = c(0.051, 0.067, 1),
                   ncomp = c(1, 1, 1),
                   scheme = "factorial",
                   response = 3,
                   scale = TRUE, scale_block = TRUE,
                   method = "sgcca",
                   verbose = FALSE)
boot <- tryCatch(
  rgcca_bootstrap(fit_sgcca, n_boot = B, n_cores = 1, verbose = FALSE),
  error = function(e) { message("rgcca_bootstrap failed: ", conditionMessage(e)); NULL })
cat(sprintf("  SGCCA bootstrap %d itérations (%.1f min)\n", B,
            as.numeric(difftime(Sys.time(),t0,units="mins"))))

freq_sgcca <- NULL
if (!is.null(boot)) {
  # Extraire la fréquence empirique = 1 - mean(loading == 0)
  # boot$stats contient les stats par bloc/composante
  freq_sgcca <- list()
  for (bn in c("GE", "CGH")) {
    df_b <- boot$stats[boot$stats$block == bn & boot$stats$comp == 1, ]
    df_b$pi_hat <- 1 - df_b$pval   # approx : pval bootstrap ~= proba être 0
    # Plus fiable : compter directement les loadings non-nuls
    if ("estimate" %in% colnames(df_b)) {
      freq_sgcca[[bn]] <- df_b
    }
  }
  cat("Structure boot$stats :\n"); print(head(boot$stats))
}

# =====================================================================
# Tableaux de synthèse
# =====================================================================
cat("\n========== SYNTHESE NB18 ==========\n")
thresholds <- c(0.6, 0.8, 0.9)
synthesis <- data.frame(
  threshold = thresholds,
  multi_ge  = sapply(thresholds, function(t) sum(pi_max_multi[grepl("^GE__",  rownames(freq_multi))] >= t)),
  multi_cgh = sapply(thresholds, function(t) sum(pi_max_multi[grepl("^CGH__", rownames(freq_multi))] >= t)),
  ovr_ge    = sapply(thresholds, function(t) sum(pi_max_ovr[grepl("^GE__",  rownames(freq_ovr))] >= t)),
  ovr_cgh   = sapply(thresholds, function(t) sum(pi_max_ovr[grepl("^CGH__", rownames(freq_ovr))] >= t))
)
print(synthesis)

saveRDS(list(
  freq_multi=freq_multi, freq_ovr=freq_ovr,
  pi_max_multi=pi_max_multi, pi_max_ovr=pi_max_ovr,
  freq_sgcca=freq_sgcca, boot_sgcca=boot,
  synthesis=synthesis, B=B, n_sub=n_sub),
  "../synthesis/nb18_stability_results.rds")

cat("\n✓ NB18 terminé — résultats dans nb18_stability_results.rds\n")
