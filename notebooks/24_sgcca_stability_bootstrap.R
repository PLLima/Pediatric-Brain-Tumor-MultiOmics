# ==========================================================================
# 24_sgcca_stability_bootstrap.R
# Stabilité de la sélection de variables de la SGCCA par bootstrap.
#
# On fige la sparsité à l'optimum CV (Set 8 : s_GE = 0.051, s_CGH = 0.067),
# puis pour B ré-échantillons bootstrap des n=39 patients d'entraînement on
# refit la SGCCA (ncomp=1) et on relève quelles variables sont sélectionnées
# (a != 0). La fréquence de sélection Pi_hat_k joue le rôle de la probabilité
# de sélection de la sélection par stabilité (Meinshausen & Buhlmann, 2010).
#
# Sorties :
#   synthesis/nb24_sgcca_stability_results.rds
#   synthesis/figures/fig12_sgcca_stability.{png,pdf}
# ==========================================================================

suppressPackageStartupMessages({
  library(RGCCA)
  library(ggplot2)
})

set.seed(0)

S_GE  <- 0.051   # sparsité optimale GE  (Set 8)
S_CGH <- 0.067   # sparsité optimale CGH (Set 8)
B     <- 200     # nombre de ré-échantillons bootstrap
THR   <- 0.6     # seuil de stabilité pi_thr

# --------------------------------------------------------------------------
# 1. Chargement des données (mêmes loaders que 08_rgcca_sgcca.R)
# --------------------------------------------------------------------------
DATA_DIR <- file.path(dirname(getwd()), "data")
if (!dir.exists(DATA_DIR)) DATA_DIR <- "data"
if (!dir.exists(DATA_DIR)) DATA_DIR <- "../data"
LABEL_ORDER <- c("cort", "dipg", "midl")

read_block <- function(block, split) {
  path <- file.path(DATA_DIR, sprintf("ge_cgh_locIGR__multiblocks__%s__%s.csv", block, split))
  df <- read.csv(path, stringsAsFactors = FALSE)
  rownames(df) <- df$row_id; df$row_id <- NULL
  df[] <- lapply(df, function(c) as.numeric(gsub(",", ".", as.character(c))))
  as.matrix(df)
}
read_targets <- function(split) {
  path <- file.path(DATA_DIR, sprintf("ge_cgh_locIGR__multiblocks__y__%s.csv", split))
  df <- read.csv(path, stringsAsFactors = FALSE)
  rownames(df) <- df$row_id; df$row_id <- NULL
  apply(df[, LABEL_ORDER], 1, function(row) LABEL_ORDER[which.max(row)])
}

X_GE  <- read_block("GE",  "train")
X_CGH <- read_block("CGH", "train")
y     <- factor(read_targets("train"), levels = LABEL_ORDER)

# Imputation médiane (sur le train complet)
fill_median <- function(M) {
  med <- apply(M, 2, median, na.rm = TRUE)
  for (j in seq_len(ncol(M))) M[is.na(M[, j]), j] <- med[j]
  M
}
X_GE  <- fill_median(X_GE)
X_CGH <- fill_median(X_CGH)

n  <- nrow(X_GE)
p1 <- ncol(X_GE); p2 <- ncol(X_CGH)
cat(sprintf("n=%d  GE=%d  CGH=%d\n", n, p1, p2))
cat("Classes :\n"); print(table(y))

# --------------------------------------------------------------------------
# 2. Fit de référence (échantillon complet)
# --------------------------------------------------------------------------
fit_sgcca <- function(ge, cgh, yy) {
  rgcca(blocks   = list(GE = ge, CGH = cgh, y = yy),
        response = 3, method = "sgcca",
        sparsity = c(S_GE, S_CGH, 1), ncomp = 1,
        verbose = FALSE)
}

cat("\n=== Fit de référence ===\n")
ref <- fit_sgcca(X_GE, X_CGH, y)
ref_ge  <- which(ref$a$GE[, 1]  != 0)
ref_cgh <- which(ref$a$CGH[, 1] != 0)
cat(sprintf("Référence : GE=%d/%d sélectionnées   CGH=%d/%d sélectionnées\n",
            length(ref_ge), p1, length(ref_cgh), p2))

# --------------------------------------------------------------------------
# 3. Boucle bootstrap
# --------------------------------------------------------------------------
cat(sprintf("\n=== Bootstrap (B=%d) ===\n", B))
sel_GE  <- numeric(p1)   # compteurs de sélection
sel_CGH <- numeric(p2)
n_ok <- 0
t0 <- Sys.time()
for (b in seq_len(B)) {
  idx <- sample.int(n, n, replace = TRUE)
  # au moins les 3 classes présentes pour que l'encodage one-hot / LDA tienne
  if (length(unique(y[idx])) < 3) next
  # ID uniques : RGCCA refuse les rownames dupliqués (inévitables en bootstrap)
  uid    <- sprintf("b%03d", seq_len(n))
  ge_b   <- X_GE [idx, , drop = FALSE]; rownames(ge_b)  <- uid
  cgh_b  <- X_CGH[idx, , drop = FALSE]; rownames(cgh_b) <- uid
  y_b    <- factor(y[idx], levels = LABEL_ORDER); names(y_b) <- uid
  res_b <- tryCatch({
    fb <- fit_sgcca(ge_b, cgh_b, y_b)
    list(ge = as.numeric(abs(fb$a$GE[, 1]) > 0),
         cgh = as.numeric(abs(fb$a$CGH[, 1]) > 0))
  }, error = function(e) NULL)
  if (!is.null(res_b)) {
    sel_GE  <- sel_GE  + res_b$ge
    sel_CGH <- sel_CGH + res_b$cgh
    n_ok <- n_ok + 1
  }
  if (b %% 20 == 0)
    cat(sprintf("  %d/%d (ok=%d)  %.1f min\n", b, B, n_ok,
                as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}
cat(sprintf("Bootstrap terminé : %d/%d fits valides en %.1f min\n",
            n_ok, B, as.numeric(difftime(Sys.time(), t0, units = "mins"))))

freq_GE  <- sel_GE  / n_ok
freq_CGH <- sel_CGH / n_ok
names(freq_GE)  <- colnames(X_GE)
names(freq_CGH) <- colnames(X_CGH)

# --------------------------------------------------------------------------
# 4. Statistiques de synthèse
# --------------------------------------------------------------------------
cat("\n=== Synthèse ===\n")
cat(sprintf("GE  : %d variables avec freq >= %.2f (stable set)\n", sum(freq_GE  >= THR), THR))
cat(sprintf("CGH : %d variables avec freq >= %.2f (stable set)\n", sum(freq_CGH >= THR), THR))
cat(sprintf("GE  : freq max = %.3f   freq médiane (sélection réf.) = %.3f\n",
            max(freq_GE), median(freq_GE[ref_ge])))
cat(sprintf("CGH : freq max = %.3f\n", max(freq_CGH)))

ge_top  <- sort(freq_GE,  decreasing = TRUE)[1:min(25, p1)]
cgh_top <- sort(freq_CGH, decreasing = TRUE)[1:min(25, p2)]
cat("\nTop 15 GE :\n");  print(round(head(ge_top, 15), 3))
cat("\nTop 15 CGH :\n"); print(round(head(cgh_top, 15), 3))

res <- list(freq_GE = freq_GE, freq_CGH = freq_CGH,
            ref_ge = ref_ge, ref_cgh = ref_cgh,
            B = B, n_ok = n_ok, thr = THR,
            s_ge = S_GE, s_cgh = S_CGH)
syn_dir <- file.path(dirname(getwd()), "synthesis")
if (!dir.exists(syn_dir)) syn_dir <- "synthesis"
saveRDS(res, file.path(syn_dir, "nb24_sgcca_stability_results.rds"))

# --------------------------------------------------------------------------
# 5. Figure (style fig9 : deux panneaux GE / CGH, barres horizontales)
# --------------------------------------------------------------------------
mk_df <- function(freqs, block, k) {
  top <- sort(freqs, decreasing = TRUE)[1:min(k, length(freqs))]
  data.frame(var = factor(names(top), levels = rev(names(top))),
             freq = as.numeric(top), block = block)
}
df <- rbind(mk_df(freq_GE, "GE", 25), mk_df(freq_CGH, "CGH", 25))
df$block <- factor(df$block, levels = c("GE", "CGH"))

p <- ggplot(df, aes(x = freq, y = var, fill = block)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.2f", freq)), hjust = -0.15, size = 2.6) +
  geom_vline(xintercept = THR, linetype = "dashed", colour = "grey40") +
  facet_wrap(~ block, scales = "free_y") +
  scale_fill_manual(values = c(GE = "#2E6FB7", CGH = "#C0392B")) +
  scale_x_continuous(limits = c(0, 1.05), expand = c(0, 0)) +
  labs(title = "Stabilité de la sélection de variables SGCCA — bootstrap",
       subtitle = sprintf("s_GE=%.3f, s_CGH=%.3f ; B=%d ré-échantillons ; seuil %.1f",
                          S_GE, S_CGH, res$n_ok, THR),
       x = expression(hat(Pi)[k]~"(fréquence de sélection)"), y = NULL) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none",
        panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size = 6.5),
        strip.text = element_text(face = "bold"))

fig_dir <- file.path(syn_dir, "figures")
ggsave(file.path(fig_dir, "fig12_sgcca_stability.png"), p, width = 11, height = 6, dpi = 150)
ggsave(file.path(fig_dir, "fig12_sgcca_stability.pdf"), p, width = 11, height = 6)
cat(sprintf("\nFigure : %s\n", file.path(fig_dir, "fig12_sgcca_stability.png")))
cat("=== TERMINÉ ===\n")
