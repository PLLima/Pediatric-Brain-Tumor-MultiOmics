# Recovery : re-run test step from NB17 with only valid schemes
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
X_train <- cbind(GE_tr, CGH_tr); X_test <- cbind(GE_te, CGH_te)
colnames(X_train)[1:ncol(GE_tr)] <- paste0("GE__", colnames(GE_tr))
colnames(X_train)[(ncol(GE_tr)+1):ncol(X_train)] <- paste0("CGH__", colnames(CGH_tr))
colnames(X_test) <- colnames(X_train)

# Schémas de poids (sans effective_num)
make_weights <- function(y, scheme) {
  n <- length(y); n_c <- table(y); K <- length(n_c)
  if (scheme == "none")           return(rep(1, n))
  if (scheme == "inv_prevalence") w_c <- n / (K * n_c)
  if (scheme == "sqrt_inv")       w_c <- sqrt(n / n_c)
  unname(w_c[as.character(y)])
}

# CV results déjà obtenus dans NB17
cv_results <- data.frame(
  scheme=rep(c("none","inv_prevalence","sqrt_inv"), each=2),
  family=rep(c("multinomial","ovr"), 3),
  mean_bal_acc=c(0.8423,0.8317, 0.8296,0.8275, 0.8377,0.8258),
  sd_bal_acc=c(0.1332,0.1416, 0.1179,0.1126, 0.1163,0.1423),
  mean_midl_recall=c(0.4524,0.4048, 0.5476,0.4286, 0.4048,0.4048),
  sd_midl_recall=c(0.4976,0.4904, 0.4976,0.4818, 0.4904,0.4904),
  n_folds=21)

# Test refit
test_results <- data.frame(); cms <- list()
for (sch in c("none","inv_prevalence","sqrt_inv")) {
  for (fam in c("multinomial","ovr")) {
    cat(sprintf("Test : %s | %s\n", sch, fam))
    w_full <- make_weights(y_train, sch)
    if (fam == "multinomial") {
      set.seed(SEED)
      fit <- cv.glmnet(X_train, y_train, family="multinomial",
                       type.multinomial="ungrouped", alpha=1, nfolds=10,
                       weights=w_full, standardize=TRUE)
      pred <- predict(fit, X_test, s="lambda.min", type="class")[,1]
    } else {
      classes <- levels(y_train)
      probs <- matrix(0, nrow(X_test), length(classes)); colnames(probs) <- classes
      for (k in seq_along(classes)) {
        y_k <- as.numeric(y_train == classes[k])
        set.seed(SEED)
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
print(test_results)
saveRDS(list(cv=cv_results, test=test_results, cms=cms),
        "../synthesis/nb17_class_weights_results.rds")
cat("✓ Sauvegarde nb17_class_weights_results.rds\n")
