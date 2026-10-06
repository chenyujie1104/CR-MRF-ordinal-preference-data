# =====================================================================
# MovieLens 1M example: covariate CR-MRF vs covariate Plackett-Luce.
#
# p = 5 movies chosen to share an audience, so every reviewer rated all
# five and the response matrix has no missing cells.  Reviewer age and
# gender drive the individual-specific node and edge parameters through
# x_tilde_i = c(1, female_i, age_std_i).
#
# Three outputs, all written to this directory:
#   node_slopes_MLE_vs_MPLE.txt            node slopes, joint MLE vs MPLE
#   P1_comparison_by_gender_selected5.pdf  pairwise-preference heatmaps
#   ..._selected5_random_tie_break.pdf     the same, with the Plackett-Luce
#                                          baseline refitted on randomly
#                                          tie-broken rankings
#
# Run from this directory (Application/MovieLens/):
#   Rscript example_run.R
# =====================================================================
suppressPackageStartupMessages({
  library(Rcpp)
  library(RcppArmadillo)
  library(ggplot2)
  library(doParallel)
})

registerDoParallel(1)                 # ordinalNet's foreach backend

# Paths resolve relative to this script's directory; outputs land here too.
ROOT <- normalizePath("../..")        # repository root
DATA <- file.path(ROOT, "Application", "MovieLens", "Movie1M")
Rcpp::sourceCpp(file.path(ROOT, "src", "count_graphical_individual_parallel.cpp"))
source(file.path(ROOT, "R", "count_graphical_individual_model.R"))

# ---------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------
# check.names = FALSE: the rating columns are named by bare movie ID
# ("1307", "2791", ...), which R would otherwise mangle to X1307, X2791.
dat <- read.csv(file.path(DATA, "selected5_reviewer_matrix.csv"),
                check.names = FALSE, encoding = "UTF-8")
key <- read.csv(file.path(DATA, "selected5_movies.csv"),
                check.names = FALSE, encoding = "UTF-8")

movie_cols <- as.character(key$item_id)   # 5 nodes, in selection order
cov_cols   <- c("age", "gender")
M <- 4L                                   # stars 1..5 -> levels 0..4

# Short movie labels: drop the "(year)" suffix and move a trailing
# ", The" to the front, so the titles fit a 5-column axis and table.
node_lab <- sub(" \\(\\d{4}\\)$", "", key$title)
node_lab <- sub("^(.*), The$", "The \\1", node_lab)

# ---------------------------------------------------------------------
# 2. Model matrices
# ---------------------------------------------------------------------
# The CR-MRF expects ordinal levels starting at 0; MovieLens stars at 1.
Y <- as.matrix(dat[, movie_cols])
storage.mode(Y) <- "integer"
Y <- Y - 1L

# A fixed split: first 170 reviewers train, the rest are held out.
train_idx <- 1:170
test_idx  <- (max(train_idx) + 1):nrow(Y)
Y_obs     <- Y[train_idx, ]

X_raw  <- dat[train_idx, cov_cols]
gender <- ifelse(X_raw$gender == "F", 1, 0)     # 1 = female

# Standardize age so its slope is per standard deviation.
age_mean <- mean(X_raw$age)
age_sd   <- sd(X_raw$age)
age_std  <- (X_raw$age - age_mean) / age_sd

X_tilde <- cbind(1, gender, age_std)   # x_tilde_i = c(1, x_i)
p  <- ncol(Y)                          # 5 nodes
q  <- ncol(X_raw)                      # 2 covariates
q1 <- q + 1L                           # covariate slices, intercept included

# ---------------------------------------------------------------------
# 3. Joint MLE by proximal gradient ascent
# ---------------------------------------------------------------------
Theta_node_init <- array(-0.1, dim = c(p, M, q1))
Theta_edge_init <- array(-0.1 * (1 - diag(p)), dim = c(p, p, q1))

cat(sprintf("\n=== joint MLE | n=%d  p=%d  M=%d  q=%d ===\n",
            nrow(Y_obs), p, M, q))
t0 <- Sys.time()
fit <- proximal_grad_ascent_individual_parallel_cpp(
  Theta_node_init, Theta_edge_init, X_tilde, Y_obs,
  N         = 10000L,   # importance-sampling draws per individual per iter
  step_size = 0.1,
  max_iter  = 3000L,
  lambda    = 0.0,      # 0 = unpenalized joint MLE
  verbose   = TRUE
)
cat(sprintf("Done in %.1fs (iter = %d, conv = %.6f)\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs")),
            fit$iter, fit$conv))

# ---------------------------------------------------------------------
# 4. Simulate held-out reviewers from the fitted model
# ---------------------------------------------------------------------
X_test       <- dat[test_idx, cov_cols]
gender_test  <- ifelse(X_test$gender == "F", 1, 0)
age_std_test <- (X_test$age - age_mean) / age_sd
X_test_tilde <- cbind(1, gender_test, age_std_test)

n_test  <- nrow(X_test_tilde)
K       <- 500L     # simulated rating vectors per test reviewer
n_gibbs <- 2000L    # Gibbs sweeps per chain (burn-in)

# Repeat each test row K times -> K independent chains per reviewer.
idx   <- rep(seq_len(n_test), each = K)
draws <- gibbs_cr_mrf_individual_parallel_cpp(
  fit$Theta_node, fit$Theta_edge,
  X_test_tilde[idx, , drop = FALSE],
  n_iter = n_gibbs
)
draws <- draws + 1L                   # levels 0..4 -> stars 1..5
colnames(draws) <- movie_cols

# K x n_test x p: Y_sim[k, i, ] is draw k for test reviewer i.
Y_sim  <- array(draws, dim = c(K, n_test, p),
                dimnames = list(NULL, NULL, movie_cols))
Y_test <- as.matrix(dat[test_idx, movie_cols])   # observed, 1..5 stars

# ---------------------------------------------------------------------
# 5. Node-wise MPLE (ordinalNet) on the same training data
# ---------------------------------------------------------------------
# Pseudo-likelihood counterpart of the joint fit: parallel slopes and
# category-specific intercepts, so its slopes line up with the MLE's.
X_train <- cbind(gender = gender, age = age_std)
pmle <- fit_ordinalNet_individual(Y_obs, X_train, M,
                                  symmetrize = "avg", lambdaVals = 0)
beta_pmle <- pmle$intercept_slopes               # p x q

# ---------------------------------------------------------------------
# 6. Table: node slopes beta_j, joint MLE vs node-wise MPLE
# ---------------------------------------------------------------------
# Slopes are shared across rating levels in the parallel model, so the fit
# reports them collapsed: fit$beta is p x q, one column per covariate.
beta_mle <- fit$beta

num2 <- function(x) formatC(x, format = "f", digits = 2, width = 5)
slope_tab <- data.frame(
  Movie         = node_lab,
  `Female MLE`  = num2(beta_mle[,  1]),
  `Female MPLE` = num2(beta_pmle[, 1]),
  `Age MLE`     = num2(beta_mle[,  2]),
  `Age MPLE`    = num2(beta_pmle[, 2]),
  check.names   = FALSE
)

slope_txt <- c(
  "Estimated node slopes beta_j for MovieLens 1M",
  sprintf("n_train = %d, n_test = %d, p = %d, M = %d. Age is standardised.",
          nrow(Y_obs), n_test, p, M),
  "Joint MLE (CR-MRF) vs node-wise MPLE (ordinalNet).",
  "",
  capture.output(print(slope_tab, row.names = FALSE))
)
writeLines(slope_txt, "node_slopes_MLE_vs_MPLE.txt")
cat(slope_txt, sep = "\n")
cat("\n")

# ---------------------------------------------------------------------
# 7. Covariate Plackett-Luce baseline
# ---------------------------------------------------------------------
# Reviewer i ranks the movies by log-worth
#   eta_ij = beta_j + gamma_j * female_i + delta_j * age_std_i,
# with the last movie's parameters fixed at 0 for identifiability.
unpack_parameters <- function(theta, J) {
  p <- J - 1
  list(beta  = c(theta[1:p],             0),
       gamma = c(theta[(p + 1):(2 * p)], 0),
       delta = c(theta[(2 * p + 1):(3 * p)], 0))
}

neg_loglik <- function(theta, ranking_list, gender, age_std, J) {
  pars <- unpack_parameters(theta, J)
  loglik <- 0
  for (i in seq_along(ranking_list)) {
    ranking <- ranking_list[[i]]
    if (is.null(ranking)) next
    eta <- pars$beta + pars$gamma * gender[i] + pars$delta * age_std[i]
    K <- length(ranking)
    # Sequential choice: each stage's denominator is over the items left,
    # summed by log-sum-exp for numerical stability.
    for (k in 1:(K - 1)) {
      remaining <- ranking[k:K]
      m <- max(eta[remaining])
      loglik <- loglik + eta[ranking[k]] -
        (m + log(sum(exp(eta[remaining] - m))))
    }
  }
  -loglik
}

J <- p
ranking_list <- lapply(seq_len(nrow(Y_obs)),
                       function(i) order(Y_obs[i, ], decreasing = TRUE))
fit_PL <- optim(
  par = rep(0, 3 * (J - 1)), fn = neg_loglik,
  ranking_list = ranking_list, gender = gender, age_std = age_std, J = J,
  method = "BFGS", control = list(maxit = 5000, reltol = 1e-10)
)
est_PL <- unpack_parameters(fit_PL$par, J)

# Reviewer-specific log-worths for the test reviewers (n_test x p), then
# the PL pairwise probability P(j > k) = logistic(eta_j - eta_k).
eta_test <- outer(rep(1, n_test), est_PL$beta) +
            outer(gender_test,    est_PL$gamma) +
            outer(age_std_test,   est_PL$delta)
colnames(eta_test) <- movie_cols

P_PL_i <- array(NA_real_, dim = c(n_test, p, p),
                dimnames = list(NULL, movie_cols, movie_cols))
for (j in 1:p) for (k in 1:p) if (j != k)
  P_PL_i[, j, k] <- plogis(eta_test[, j] - eta_test[, k])

# ---------------------------------------------------------------------
# 8. Pairwise preference probabilities and Frobenius errors
# ---------------------------------------------------------------------
# P[j, k] = P(Y_j > Y_k) + 0.5 P(Y_j == Y_k): ties split in half, so the
# matrix is the same object for observed data, CR-MRF draws and PL.
pair_prob <- function(Y) {
  p <- ncol(Y)
  P <- matrix(NA_real_, p, p, dimnames = list(colnames(Y), colnames(Y)))
  for (j in 1:p) for (k in 1:p) if (j != k)
    P[j, k] <- mean(Y[, j] > Y[, k]) + 0.5 * mean(Y[, j] == Y[, k])
  P
}

# Population level, pooled over the held-out reviewers.
P1_test <- pair_prob(Y_test)
P1_CR   <- pair_prob(draws)            # all K * n_test simulated vectors
P_PL    <- apply(P_PL_i, c(2, 3), mean)

# Errors are computed BEFORE the figure and pasted into the panel titles,
# so the numbers in the plot can never drift from the numbers reported.
diag(P1_test) <- diag(P1_CR) <- diag(P_PL) <- 1
err_CR <- norm(P1_test - P1_CR, "F")
err_PL <- norm(P1_test - P_PL,  "F")
cat(sprintf("Column-order tie-break | held-out Frobenius error: CR-MRF %.3f | Plackett-Luce %.3f\n",
            err_CR, err_PL))

# Same three matrices within each gender: observed rows, that group's
# pooled CR-MRF draws, and their reviewer-specific PL probabilities.
groups <- list(Male   = which(gender_test == 0),
               Female = which(gender_test == 1))

pair_prob_by_group <- function(sel) {
  Ys <- matrix(Y_sim[, sel, , drop = FALSE], ncol = p)   # collapse K x n
  colnames(Ys) <- movie_cols
  list(test = pair_prob(Y_test[sel, , drop = FALSE]),
       CR   = pair_prob(Ys),
       PL   = apply(P_PL_i[sel, , , drop = FALSE], c(2, 3), mean))
}
P_by_gender <- lapply(groups, pair_prob_by_group)

err_gender <- do.call(rbind, lapply(names(groups), function(g) {
  P <- P_by_gender[[g]]
  fix_diag <- function(A) { diag(A) <- 1; A }
  Pt <- fix_diag(P$test); Pc <- fix_diag(P$CR); Pp <- fix_diag(P$PL)
  data.frame(Gender = g, n = length(groups[[g]]),
             CR_MRF = norm(Pt - Pc, "F"),
             PL     = norm(Pt - Pp, "F"))
}))
print(err_gender, row.names = FALSE)

# ---------------------------------------------------------------------
# 9. Figure: pairwise-preference heatmaps, by gender
# ---------------------------------------------------------------------
# Movies are ordered by observed strength, the same order in every panel,
# which turns "is the model right?" into "does the gradient look alike?".
strength <- rowMeans(`diag<-`(P1_test, NA), na.rm = TRUE)
ord      <- order(strength, decreasing = TRUE)
lab_ord  <- node_lab[ord]

to_long <- function(Mat, panel) {
  Mat <- Mat[ord, ord]
  diag(Mat) <- NA                  # self-comparisons carry no information
  data.frame(
    row   = factor(rep(lab_ord, times = p), levels = lab_ord),
    col   = factor(rep(lab_ord, each  = p), levels = lab_ord),
    prob  = as.vector(Mat),        # column-major: row index varies fastest
    panel = panel
  )
}

# One panel per (gender, source); `g` carries the scales and theme, and
# the data are swapped in below with `%+%`.
d_gender <- do.call(rbind, lapply(names(groups), function(gg) {
  P  <- P_by_gender[[gg]]
  e  <- err_gender[err_gender$Gender == gg, ]
  pn <- c(sprintf("%s (n = %d): observed",               gg, e$n),
          sprintf("%s: Plackett-Luce  (error %.3f)",      gg, e$PL),
          sprintf("%s: CR-MRF  (error %.3f)",             gg, e$CR_MRF))
  rbind(to_long(P$test, pn[1]),
        to_long(P$PL,   pn[2]),
        to_long(P$CR,   pn[3]))
}))
d_gender$panel <- factor(d_gender$panel, levels = unique(d_gender$panel))

g <- ggplot(d_gender, aes(col, row, fill = prob)) +
  geom_tile(colour = "white", linewidth = 0.7) +
  geom_text(data = function(x) subset(x, !is.na(prob)),
            aes(label = sprintf("%.2f", prob), colour = abs(prob - 0.5) > 0.33),
            size = 3, show.legend = FALSE) +
  scale_colour_manual(values = c(`FALSE` = "grey15", `TRUE` = "white")) +
  # Diverging, centred on 0.5: white is a coin flip, red = row wins,
  # blue = row loses. A sequential ramp hides that 0.5 is the null value.
  scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                       midpoint = 0.5, limits = c(0, 1),
                       breaks = c(0, 0.25, 0.5, 0.75, 1),
                       labels = c("0", ".25", ".5 (tie)", ".75", "1"),
                       na.value = "grey92",
                       name = "P(row preferred to column) + 0.5 P(tie)") +
  scale_y_discrete(limits = rev) +
  coord_equal() +
  facet_wrap(~ panel, nrow = 2) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid    = element_blank(),
    axis.text.x   = element_text(angle = 30, hjust = 1),
    axis.ticks    = element_blank(),
    strip.text    = element_text(face = "bold", size = 11, margin = margin(b = 6)),
    legend.position       = "bottom",
    legend.key.width      = grid::unit(2.4, "cm"),
    legend.key.height     = grid::unit(0.35, "cm"),
    legend.title.position = "top",
    plot.margin   = margin(6, 10, 6, 6)
  )

ggsave("P1_comparison_by_gender_selected5.pdf", g, width = 11.5, height = 9.5)

# ---------------------------------------------------------------------
# 10. Plackett-Luce refit with random tie-breaking
# ---------------------------------------------------------------------
set.seed(19961104); B <- 10
ranking_list <- unlist(lapply(seq_len(B), function(b)
                         lapply(seq_len(nrow(Y_obs)), function(i)
                           order(Y_obs[i, ], sample(p), decreasing = TRUE))),
                       recursive = FALSE)
gender_r <- rep(gender, B); age_r <- rep(age_std, B)

# Summing the log-likelihood over the B copies averages it over the
# tie-breakings, up to the factor B, which does not move the optimum.
fit_PL <- optim(
  par = rep(0, 3 * (J - 1)), fn = neg_loglik,
  ranking_list = ranking_list, gender = gender_r, age_std = age_r, J = J,
  method = "BFGS", control = list(maxit = 5000, reltol = 1e-10)
)

est_PL <- unpack_parameters(fit_PL$par, J)

eta_test <- outer(rep(1, n_test), est_PL$beta) +
            outer(gender_test,    est_PL$gamma) +
            outer(age_std_test,   est_PL$delta)
colnames(eta_test) <- movie_cols

P_PL_i <- array(NA_real_, dim = c(n_test, p, p),
                dimnames = list(NULL, movie_cols, movie_cols))
for (j in 1:p) for (k in 1:p) if (j != k)
  P_PL_i[, j, k] <- plogis(eta_test[, j] - eta_test[, k])

P1_test <- pair_prob(Y_test)
P1_CR   <- pair_prob(draws)
P_PL    <- apply(P_PL_i, c(2, 3), mean)

diag(P1_test) <- diag(P1_CR) <- diag(P_PL) <- 1
err_CR <- norm(P1_test - P1_CR, "F")
err_PL <- norm(P1_test - P_PL,  "F")
cat(sprintf("Random tie-break      | held-out Frobenius error: CR-MRF %.3f | Plackett-Luce %.3f\n",
            err_CR, err_PL))

groups <- list(Male   = which(gender_test == 0),
               Female = which(gender_test == 1))

pair_prob_by_group <- function(sel) {
  Ys <- matrix(Y_sim[, sel, , drop = FALSE], ncol = p)   # collapse K x n
  colnames(Ys) <- movie_cols
  list(test = pair_prob(Y_test[sel, , drop = FALSE]),
       CR   = pair_prob(Ys),
       PL   = apply(P_PL_i[sel, , , drop = FALSE], c(2, 3), mean))
}
P_by_gender <- lapply(groups, pair_prob_by_group)

err_gender <- do.call(rbind, lapply(names(groups), function(g) {
  P <- P_by_gender[[g]]
  fix_diag <- function(A) { diag(A) <- 1; A }
  Pt <- fix_diag(P$test); Pc <- fix_diag(P$CR); Pp <- fix_diag(P$PL)
  data.frame(Gender = g, n = length(groups[[g]]),
             CR_MRF = norm(Pt - Pc, "F"),
             PL     = norm(Pt - Pp, "F"))
}))
print(err_gender, row.names = FALSE)

# ---------------------------------------------------------------------
# 11. Figure: the same heatmaps under random tie-breaking
# ---------------------------------------------------------------------

strength <- rowMeans(`diag<-`(P1_test, NA), na.rm = TRUE)
ord      <- order(strength, decreasing = TRUE)
lab_ord  <- node_lab[ord]

to_long <- function(Mat, panel) {
  Mat <- Mat[ord, ord]
  diag(Mat) <- NA                  # self-comparisons carry no information
  data.frame(
    row   = factor(rep(lab_ord, times = p), levels = lab_ord),
    col   = factor(rep(lab_ord, each  = p), levels = lab_ord),
    prob  = as.vector(Mat),        # column-major: row index varies fastest
    panel = panel
  )
}

d_gender <- do.call(rbind, lapply(names(groups), function(gg) {
  P  <- P_by_gender[[gg]]
  e  <- err_gender[err_gender$Gender == gg, ]
  pn <- c(sprintf("%s (n = %d): observed",               gg, e$n),
          sprintf("%s: Plackett-Luce  (error %.3f)",      gg, e$PL),
          sprintf("%s: CR-MRF  (error %.3f)",             gg, e$CR_MRF))
  rbind(to_long(P$test, pn[1]),
        to_long(P$PL,   pn[2]),
        to_long(P$CR,   pn[3]))
}))
d_gender$panel <- factor(d_gender$panel, levels = unique(d_gender$panel))

g <- ggplot(d_gender, aes(col, row, fill = prob)) +
  geom_tile(colour = "white", linewidth = 0.7) +
  geom_text(data = function(x) subset(x, !is.na(prob)),
            aes(label = sprintf("%.2f", prob), colour = abs(prob - 0.5) > 0.33),
            size = 3, show.legend = FALSE) +
  scale_colour_manual(values = c(`FALSE` = "grey15", `TRUE` = "white")) +
  # Diverging, centred on 0.5: white is a coin flip, red = row wins,
  # blue = row loses. A sequential ramp hides that 0.5 is the null value.
  scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                       midpoint = 0.5, limits = c(0, 1),
                       breaks = c(0, 0.25, 0.5, 0.75, 1),
                       labels = c("0", ".25", ".5 (tie)", ".75", "1"),
                       na.value = "grey92",
                       name = "P(row preferred to column) + 0.5 P(tie)") +
  scale_y_discrete(limits = rev) +
  coord_equal() +
  facet_wrap(~ panel, nrow = 2) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid    = element_blank(),
    axis.text.x   = element_text(angle = 30, hjust = 1),
    axis.ticks    = element_blank(),
    strip.text    = element_text(face = "bold", size = 11, margin = margin(b = 6)),
    legend.position       = "bottom",
    legend.key.width      = grid::unit(2.4, "cm"),
    legend.key.height     = grid::unit(0.35, "cm"),
    legend.title.position = "top",
    plot.margin   = margin(6, 10, 6, 6)
  )

ggsave("P1_comparison_by_gender_selected5_random_tie_break.pdf", g,
       width = 11.5, height = 9.5)
