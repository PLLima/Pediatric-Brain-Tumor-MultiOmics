# =====================================================================
# NB00 — Analyse Exploratoire des Données (EDA)
# =====================================================================
# Génère :
#   - eda_class_distribution.png : barplot des 3 classes train/test
#   - eda_ge_distribution.png    : distribution log-expression GE
#   - eda_cgh_profile.png        : profil moyen CGH par classe
#   - eda_ge_top_var.png         : top-30 GE par variance, heatmap
#   - eda_pca_blocks.png         : PCA non-supervisée GE / CGH
#   - eda_pca_scree.png          : scree + variance cumulée PCA par bloc
#   - eda_ge_top_discriminants.png : top-20 GE par |t-stat| vs midl
# =====================================================================

suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(reshape2)
})

set.seed(42)
LABEL_ORDER <- c("cort", "dipg", "midl")
COL_CLASSES <- c("cort"="#3498DB", "dipg"="#F39C12", "midl"="#C0392B")

# ---- Localiser data (dossier du script, portable Windows/macOS/Linux) ----
get_script_dir <- function() {
  # 1) Rscript / Code Runner : argument --file=
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f)) return(dirname(normalizePath(f[1])))
  # 2) source() interactif
  if (!is.null(sys.function(1)) && !is.null(ofn <- sys.frame(1)$ofile))
    return(dirname(normalizePath(ofn)))
  # 3) repli : répertoire courant
  getwd()
}
script_dir <- tryCatch(get_script_dir(), error = function(e) getwd())
setwd(script_dir)
data_dir <- "../data"
OUT_DIR  <- "../synthesis/figures"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- Helpers ----
to_numeric_frame <- function(df) {
  rn <- rownames(df)
  out <- as.data.frame(
    lapply(df, function(x) as.numeric(gsub(",", ".", as.character(x), fixed=TRUE))),
    check.names = FALSE)
  rownames(out) <- rn; out
}
load_block <- function(blk, split) {
  df <- as.data.frame(fread(file.path(data_dir,
    sprintf("ge_cgh_locIGR__multiblocks__%s__%s.csv", blk, split)),
    check.names = FALSE))
  rownames(df) <- as.character(df[[1]]); df[[1]] <- NULL
  to_numeric_frame(df)
}
load_y <- function(split) {
  df <- as.data.frame(fread(file.path(data_dir,
    sprintf("ge_cgh_locIGR__multiblocks__y__%s.csv", split)),
    check.names = FALSE))
  rownames(df) <- as.character(df[[1]]); df[[1]] <- NULL
  factor(LABEL_ORDER[max.col(as.matrix(df[, LABEL_ORDER]), ties.method = "first")],
         levels = LABEL_ORDER)
}

GE_tr_raw  <- load_block("GE",  "train")
GE_te_raw  <- load_block("GE",  "test")
CGH_tr_raw <- load_block("CGH", "train")
CGH_te_raw <- load_block("CGH", "test")
y_tr_raw   <- load_y("train"); y_te_raw <- load_y("test")

# Aligner par intersection des IDs (comme dans NB09/NB11/NB14)
y_tr_path <- file.path(data_dir, "ge_cgh_locIGR__multiblocks__y__train.csv")
y_te_path <- file.path(data_dir, "ge_cgh_locIGR__multiblocks__y__test.csv")
y_tr_ids <- as.character(as.data.frame(fread(y_tr_path))[[1]])
y_te_ids <- as.character(as.data.frame(fread(y_te_path))[[1]])
names(y_tr_raw) <- y_tr_ids
names(y_te_raw) <- y_te_ids

train_ids <- Reduce(intersect, list(rownames(GE_tr_raw),
                                     rownames(CGH_tr_raw),
                                     names(y_tr_raw)))
test_ids  <- Reduce(intersect, list(rownames(GE_te_raw),
                                     rownames(CGH_te_raw),
                                     names(y_te_raw)))

GE_tr  <- as.matrix(GE_tr_raw [train_ids, , drop = FALSE])
CGH_tr <- as.matrix(CGH_tr_raw[train_ids, , drop = FALSE])
y_tr   <- y_tr_raw[train_ids]
GE_te  <- as.matrix(GE_te_raw [test_ids,  , drop = FALSE])
CGH_te <- as.matrix(CGH_te_raw[test_ids,  , drop = FALSE])
y_te   <- y_te_raw[test_ids]

# Impute median (train stats)
impute_median <- function(tr, te) {
  med <- apply(tr, 2, median, na.rm=TRUE)
  for (j in 1:ncol(tr)) {
    tr[is.na(tr[,j]), j] <- med[j]
    te[is.na(te[,j]), j] <- med[j]
  }
  list(tr=tr, te=te)
}
f <- impute_median(GE_tr, GE_te);  GE_tr <- f$tr; GE_te <- f$te
f <- impute_median(CGH_tr, CGH_te); CGH_tr <- f$tr; CGH_te <- f$te

cat(sprintf("Train: %d patients | Test: %d patients\n", length(y_tr), length(y_te)))
cat(sprintf("GE: %d features | CGH: %d features\n", ncol(GE_tr), ncol(CGH_tr)))

# =====================================================================
# 1. Distribution des classes
# =====================================================================
df_cls <- rbind(
  data.frame(set="Train", class=as.character(y_tr)),
  data.frame(set="Test",  class=as.character(y_te)))
df_cls$class <- factor(df_cls$class, levels = LABEL_ORDER)

p1 <- ggplot(df_cls, aes(x = class, fill = class)) +
  geom_bar(color = "black", linewidth = 0.4) +
  geom_text(stat = "count", aes(label = after_stat(count)),
            vjust = -0.4, size = 4, fontface = "bold") +
  facet_wrap(~ set, scales = "free_y") +
  scale_fill_manual(values = COL_CLASSES, guide = "none") +
  labs(title = "Distribution des classes (train / test)",
       x = NULL, y = "Nombre de patients") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        strip.text = element_text(face = "bold"))
ggsave(file.path(OUT_DIR, "eda_class_distribution.png"), p1,
       width = 7, height = 4, dpi = 150, bg = "white")
cat("✓ eda_class_distribution.png\n")

# =====================================================================
# 2. Distribution globale GE (échelle log)
# =====================================================================
ge_vals <- as.vector(GE_tr)
df_ge <- data.frame(expr = ge_vals)
p2 <- ggplot(df_ge, aes(x = expr)) +
  geom_histogram(bins = 80, fill = "#3498DB", color = "black",
                 linewidth = 0.2, alpha = 0.85) +
  labs(title = sprintf("Distribution globale GE (n=%d patients × %d gènes)",
                        nrow(GE_tr), ncol(GE_tr)),
       subtitle = sprintf("min = %.2f | median = %.2f | max = %.2f",
                          min(ge_vals), median(ge_vals), max(ge_vals)),
       x = "Niveau d'expression normalisé", y = "Fréquence") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(OUT_DIR, "eda_ge_distribution.png"), p2,
       width = 8, height = 4, dpi = 150, bg = "white")
cat("✓ eda_ge_distribution.png\n")

# =====================================================================
# 3. Profil CGH moyen par classe (proxy spatial : ordre numérique)
# =====================================================================
cgh_ids <- as.numeric(colnames(CGH_tr))
ord     <- order(cgh_ids)
cgh_sorted <- CGH_tr[, ord]
ids_sorted <- cgh_ids[ord]

profiles <- sapply(LABEL_ORDER, function(cl) colMeans(cgh_sorted[y_tr == cl, ]))
df_prof <- data.frame(
  segment_id = rep(ids_sorted, 3),
  classe     = factor(rep(LABEL_ORDER, each = length(ids_sorted)),
                      levels = LABEL_ORDER),
  signal     = c(profiles[,"cort"], profiles[,"dipg"], profiles[,"midl"]))

p3 <- ggplot(df_prof, aes(x = segment_id, y = signal, color = classe)) +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed", alpha = 0.4) +
  geom_line(linewidth = 0.35, alpha = 0.85) +
  facet_wrap(~ classe, ncol = 1, scales = "fixed") +
  scale_color_manual(values = COL_CLASSES, guide = "none") +
  labs(title = "Profil moyen CGH par classe (ordre génomique approximatif)",
       subtitle = "Signal > 0 : gain ; signal < 0 : perte. Adjacence numérique = proxy de l'adjacence chromosomique.",
       x = "ID segment (ordre numérique)", y = "Signal CGH moyen") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold"))
ggsave(file.path(OUT_DIR, "eda_cgh_profile.png"), p3,
       width = 10, height = 6, dpi = 150, bg = "white")
cat("✓ eda_cgh_profile.png\n")

# =====================================================================
# 4. Top-30 GE par variance — heatmap
# =====================================================================
gene_var <- apply(GE_tr, 2, var)
top30 <- order(gene_var, decreasing = TRUE)[1:30]
heat <- GE_tr[order(y_tr), top30]
ann_y <- sort(y_tr)
df_h <- melt(heat); colnames(df_h) <- c("patient", "gene", "expr")
df_h$class <- ann_y[df_h$patient]
df_h$patient <- factor(df_h$patient, levels = unique(df_h$patient))

p4 <- ggplot(df_h, aes(x = gene, y = patient, fill = expr)) +
  geom_tile() +
  scale_fill_gradient2(low = "#2C3E50", mid = "white", high = "#C0392B",
                       midpoint = 0, name = "expr") +
  geom_tile(aes(x = -1, fill = NULL, color = class), width = 1, height = 1) +
  scale_color_manual(values = COL_CLASSES, name = "Classe") +
  labs(title = "Heatmap des 30 GE de plus forte variance",
       subtitle = "Patients triés par classe (vue qualitative)",
       x = "Gène (ordre variance décroissante)", y = "Patient") +
  theme_minimal(base_size = 9) +
  theme(plot.title = element_text(face = "bold"),
        axis.text.x = element_text(angle = 90, hjust = 1, size = 6),
        axis.text.y = element_text(size = 5))
ggsave(file.path(OUT_DIR, "eda_ge_top_var.png"), p4,
       width = 9, height = 6, dpi = 150, bg = "white")
cat("✓ eda_ge_top_var.png\n")

# =====================================================================
# 5. PCA non supervisée GE et CGH
# =====================================================================
pca_plot <- function(X, ylab, fname, title_suffix) {
  X_std <- scale(X)
  X_std[is.na(X_std)] <- 0
  pc <- prcomp(X_std, center = FALSE, scale. = FALSE)
  ve <- (pc$sdev^2 / sum(pc$sdev^2)) * 100
  df <- data.frame(PC1 = pc$x[,1], PC2 = pc$x[,2], classe = y_tr)
  p <- ggplot(df, aes(x = PC1, y = PC2, color = classe, shape = classe)) +
    geom_point(size = 4, alpha = 0.85, stroke = 1) +
    stat_ellipse(level = 0.7, linewidth = 0.4, linetype = "dashed", alpha = 0.6) +
    scale_color_manual(values = COL_CLASSES) +
    labs(title = sprintf("PCA non-supervisée — %s", title_suffix),
         x = sprintf("PC1 (%.1f %%)", ve[1]),
         y = sprintf("PC2 (%.1f %%)", ve[2])) +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold"))
  ggsave(file.path(OUT_DIR, fname), p, width = 6.5, height = 5,
         dpi = 150, bg = "white")
}
pca_plot(GE_tr,  "GE",  "eda_pca_ge.png",  "Bloc GE (15 702 features)")
pca_plot(CGH_tr, "CGH", "eda_pca_cgh.png", "Bloc CGH (1 229 features)")
cat("✓ eda_pca_ge.png + eda_pca_cgh.png\n")

# =====================================================================
# 6. Top GE discriminants pour midl (t-stat absolue)
# =====================================================================
y_midl  <- as.numeric(y_tr == "midl")
t_stats <- apply(GE_tr, 2, function(x) {
  m1 <- mean(x[y_midl == 1]); m0 <- mean(x[y_midl == 0])
  s1 <- sd(x[y_midl == 1]);   s0 <- sd(x[y_midl == 0])
  n1 <- sum(y_midl); n0 <- length(y_midl) - n1
  se <- sqrt(s1^2 / n1 + s0^2 / n0)
  if (is.na(se) || se < 1e-10) return(0)
  (m1 - m0) / se
})
ord_t <- order(abs(t_stats), decreasing = TRUE)[1:20]
df_top <- data.frame(
  gene = factor(colnames(GE_tr)[ord_t],
                levels = colnames(GE_tr)[ord_t]),
  tstat = t_stats[ord_t],
  sens = ifelse(t_stats[ord_t] > 0, "Sur-exprimé midl", "Sous-exprimé midl"))

p6 <- ggplot(df_top, aes(x = gene, y = tstat, fill = sens)) +
  geom_col(color = "black", linewidth = 0.3) +
  scale_fill_manual(values = c("Sur-exprimé midl" = "#C0392B",
                                "Sous-exprimé midl" = "#3498DB"),
                    name = NULL) +
  labs(title = "Top 20 gènes discriminants pour midl (t-test univariate)",
       subtitle = "Avant tout modèle. Indique si un signal univarié existe pour la classe rare.",
       x = NULL, y = "t-statistique (Welch)") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold"),
        axis.text.x = element_text(angle = 60, hjust = 1, size = 7),
        legend.position = "top")
ggsave(file.path(OUT_DIR, "eda_ge_top_discriminants.png"), p6,
       width = 9, height = 5, dpi = 150, bg = "white")
cat("✓ eda_ge_top_discriminants.png\n")

# =====================================================================
# 7. Structure de corrélation via PCA — scree + variance cumulée
# =====================================================================
# Une matrice de corrélation brute sur des features tirés au hasard est
# illisible (15 702 / 1 229 variables) et peu interprétable. La PCA résume
# cette structure de corrélation : la vitesse à laquelle la variance se
# concentre dans les premières composantes mesure la redondance
# (collinéarité) du bloc et sa dimensionnalité effective.
scree_df <- function(X, blk) {
  X_std <- scale(X)
  X_std[is.na(X_std)] <- 0
  pc <- prcomp(X_std, center = FALSE, scale. = FALSE)
  ve <- pc$sdev^2 / sum(pc$sdev^2)
  k  <- length(ve)
  data.frame(bloc = blk, PC = seq_len(k),
             var_expl = ve * 100, var_cum = cumsum(ve) * 100)
}
df_scree <- rbind(scree_df(GE_tr, "GE"), scree_df(CGH_tr, "CGH"))
df_scree$bloc <- factor(df_scree$bloc, levels = c("GE", "CGH"))

# Nombre de PC pour atteindre 80 % / 90 % de variance par bloc
thr <- do.call(rbind, lapply(levels(df_scree$bloc), function(b) {
  d <- df_scree[df_scree$bloc == b, ]
  data.frame(bloc = b,
             k80 = which(d$var_cum >= 80)[1],
             k90 = which(d$var_cum >= 90)[1])
}))
lab_thr <- with(thr, sprintf("%s : %d PC → 80 %% | %d PC → 90 %%",
                             bloc, k80, k90))

p7 <- ggplot(df_scree, aes(x = PC)) +
  geom_col(aes(y = var_expl), fill = "#3498DB", color = "black",
           linewidth = 0.15, alpha = 0.85) +
  geom_line(aes(y = var_cum), color = "#C0392B", linewidth = 0.6) +
  geom_point(aes(y = var_cum), color = "#C0392B", size = 0.9) +
  geom_hline(yintercept = c(80, 90), linetype = "dashed",
             color = "grey40", alpha = 0.6) +
  facet_wrap(~ bloc, scales = "free_x") +
  labs(title = "Structure de corrélation par bloc — PCA (scree plot)",
       subtitle = paste(lab_thr, collapse = "   •   "),
       x = "Composante principale",
       y = "Variance expliquée (%)  —  barres ; cumulée (%) — courbe") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold"))
ggsave(file.path(OUT_DIR, "eda_pca_scree.png"), p7,
       width = 10, height = 4.5, dpi = 150, bg = "white")
cat("✓ eda_pca_scree.png\n")

# =====================================================================
# 8. Sauvegarde résumé
# =====================================================================
summary_eda <- list(
  n_train = length(y_tr), n_test = length(y_te),
  p_ge = ncol(GE_tr), p_cgh = ncol(CGH_tr),
  ratio_p_n = (ncol(GE_tr) + ncol(CGH_tr)) / length(y_tr),
  class_train = table(y_tr), class_test = table(y_te),
  ge_summary = summary(as.vector(GE_tr)),
  cgh_summary = summary(as.vector(CGH_tr)),
  top10_var_genes = colnames(GE_tr)[order(gene_var, decreasing=TRUE)[1:10]],
  top10_tstat_midl = colnames(GE_tr)[ord_t[1:10]],
  midl_intercept_OvR = log(8/31)  # pour rappel
)
saveRDS(summary_eda, file.path("..", "synthesis", "nb00_eda_summary.rds"))
cat("\n✓ EDA terminée — 7 figures + summary RDS\n")
