# =====================================================================
# CR-MRF with individual-specific parameters via a linear covariate model
#
# For each individual i with covariates x_i = (x_{i1}, ..., x_{iq}) and
# augmented covariate x_tilde_i = c(1, x_i):
#
#   theta_{s,j}(x_i) = sum_{v=0}^q  theta_{s,j,v} * x_tilde_{iv}
#   theta_{st}(x_i)  = sum_{v=0}^q  theta_{st,v}  * x_tilde_{iv}
#
# Parameter storage (3D arrays, extending the 2D matrices in the base file):
#   Theta_node : p x M x (q+1)   -- slice [,,1] = intercept, [,,v+1] = coef of x_v
#   Theta_edge : p x p x (q+1)   -- each slice symmetric with zero diagonal
#
# The base file is sourced for shared helpers (node_stat, node_pmf_indep,
# soft_threshold, sample_independent) and for .symmetrize_theta_edge used by
# fit_ordinalNet_individual().
# =====================================================================

# The calling script sets ROOT to the repository root; fall back to the
# working directory being the repository root.
if (!exists("ROOT")) ROOT <- "."
source(file.path(ROOT, "R", "count_graphical_cr_mrf.R"))

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# Prepend an intercept column to a covariate matrix X (n x q -> n x (q+1)).
augment_X <- function(X) cbind(1, as.matrix(X))

# Vectorized row-wise node_stat over a matrix of shape (n x M).
# Returns an (n x (M+1)) matrix whose row i is node_stat(Tn_s[i, ]).
node_stat_rows <- function(Tn_s) {
  M <- ncol(Tn_s)
  if (M == 0) return(matrix(0, nrow(Tn_s), 1))
  # reverse columns, cumsum along rows, reverse back, append 0 column
  rev_cs <- t(apply(Tn_s[, M:1, drop = FALSE], 1, cumsum))[, M:1, drop = FALSE]
  cbind(rev_cs, 0)
}

# ---------------------------------------------------------------------
# Build per-individual parameter arrays
# ---------------------------------------------------------------------
# Given Theta_node (p x M x (q+1)), Theta_edge (p x p x (q+1)) and the
# augmented covariate matrix X_tilde (n x (q+1)), build:
#   Tn_all : n x p x M    -- Tn_all[i, s, j] = theta_{s,j}(x_i)
#   Te_all : n x p x p    -- Te_all[i, s, t] = theta_{st}(x_i)
#
# Each edge slice in the input is assumed already symmetric / zero-diag,
# so the per-individual matrices inherit those properties.

make_individual_params <- function(Theta_node, Theta_edge, X_tilde) {
  p  <- dim(Theta_node)[1]
  M  <- dim(Theta_node)[2]
  q1 <- dim(Theta_node)[3]
  n  <- nrow(X_tilde)

  # Flatten over the (s, j) index, contract along v, reshape back.
  Tn_mat       <- matrix(Theta_node, nrow = p * M, ncol = q1)
  Tn_all_flat  <- X_tilde %*% t(Tn_mat)                       # n x (p*M)
  Tn_all       <- array(Tn_all_flat, dim = c(n, p, M))

  Te_mat       <- matrix(Theta_edge, nrow = p * p, ncol = q1)
  Te_all_flat  <- X_tilde %*% t(Te_mat)                       # n x (p*p)
  Te_all       <- array(Te_all_flat, dim = c(n, p, p))

  list(Tn_all = Tn_all, Te_all = Te_all)
}

# ---------------------------------------------------------------------
# Independence sampler with individual-specific node parameters
# ---------------------------------------------------------------------
# Tn_all is n x p x M. Returns one draw per individual: n x p integer matrix.

sample_independent_individual <- function(Tn_all) {
  n <- dim(Tn_all)[1]
  p <- dim(Tn_all)[2]
  M <- dim(Tn_all)[3]
  X <- matrix(0L, nrow = n, ncol = p)
  for (s in 1:p) {
    # Compute per-individual pmfs for node s in one shot.
    # log_unnorm[i, k+1] = node_stat(Tn_all[i, s, ])[k+1], k = 0,...,M.
    Tn_s <- matrix(Tn_all[, s, ], nrow = n, ncol = M)               # n x M
    log_unnorm <- node_stat_rows(Tn_s)                              # n x (M+1)
    log_unnorm <- log_unnorm - apply(log_unnorm, 1, max)
    probs <- exp(log_unnorm)
    probs <- probs / rowSums(probs)
    cdf <- t(apply(probs, 1, cumsum))
    u   <- runif(n)
    X[, s] <- rowSums(cdf < u)
  }
  X
}

# ---------------------------------------------------------------------
# Gibbs sampler: one draw per individual, parameters individual-specific
# ---------------------------------------------------------------------
# Returns Y of shape n x p, where row i is approximately a sample from
# p_{Theta^{(i)}}(.).
#
# Mirrors the structure of gibbs_cr_mrf in the base file: vectorized across
# the n parallel chains, looping over coordinates s. The only structural
# difference is that node parameters and edge couplings differ per chain.

gibbs_cr_mrf_individual <- function(Theta_node, Theta_edge, X_tilde,
                                    n_iter = 500) {
  pars   <- make_individual_params(Theta_node, Theta_edge, X_tilde)
  Tn_all <- pars$Tn_all                              # n x p x M
  Te_all <- pars$Te_all                              # n x p x p

  n <- dim(Tn_all)[1]
  p <- dim(Tn_all)[2]
  M <- dim(Tn_all)[3]

  X  <- sample_independent_individual(Tn_all)
  MY <- (M - X) / M                                  # n x p

  scale_vec <- (M - 0:M) / M                         # length M+1

  for (iter in 1:n_iter) {
    for (s in 1:p) {
      # r_s[i] = sum_{t != s} Te_all[i, s, t] * MY[i, t]
      Te_s <- matrix(Te_all[, s, ], nrow = n, ncol = p)
      Te_s[, s] <- 0                                 # safety: zero self-coupling
      r_s <- rowSums(Te_s * MY)                      # length n

      # Node-stat rows for node s under each individual's parameters.
      Tn_s <- matrix(Tn_all[, s, ], nrow = n, ncol = M)
      NS_s <- node_stat_rows(Tn_s)                   # n x (M+1)

      # Full-conditional log-unnormalised pmf: n x (M+1)
      log_unnorm <- NS_s + outer(r_s, scale_vec)
      log_unnorm <- log_unnorm - apply(log_unnorm, 1, max)
      probs <- exp(log_unnorm)
      probs <- probs / rowSums(probs)

      # Sample by inverse CDF across the n chains.
      cdf <- t(apply(probs, 1, cumsum))
      u   <- runif(n)
      X[, s] <- rowSums(cdf < u)
      MY[, s] <- (M - X[, s]) / M
    }
  }
  X
}

# ---------------------------------------------------------------------
# Importance sampling for one individual: log(Z_i / Z_{0,i}) and the
# moments needed by the gradient.
# ---------------------------------------------------------------------
# Inputs:
#   Tn_i : p x M        node parameters for individual i
#   Te_i : p x p        edge couplings for individual i (symmetric, zero diag)
#   N    : number of IS samples
#
# Outputs:
#   log_Z_ratio = log(Z_i / Z_{0,i})
#   E_node      = p x M, E_{p_i}[ I[Y_s <= j - 1] ]
#   E_edge      = p x p, E_{p_i}[ (M - Y_s)(M - Y_t) / M^2 ]

z_moments_individual <- function(Tn_i, Te_i, N) {
  p <- nrow(Tn_i); M <- ncol(Tn_i)

  # Sample under the independence model for this individual.
  Y <- matrix(0L, N, p)
  for (s in 1:p) {
    pmf <- node_pmf_indep(Tn_i[s, ])
    Y[, s] <- sample(0:M, N, replace = TRUE, prob = pmf)
  }

  MY <- (M - Y) / M                                  # N x p

  # q_b = 0.5 * MY_b' Te_i MY_b
  q_vals <- rowSums((MY %*% Te_i) * MY) / 2          # length N
  q_max  <- max(q_vals)
  d      <- exp(q_vals - q_max)
  sum_d  <- sum(d)

  log_Z_ratio <- q_max + log(sum_d / N)

  # Node moments: E[ I[Y_s <= j - 1] ]
  E_node <- matrix(0, p, M)
  for (jj in 1:M) {
    indicators <- (Y <= (jj - 1)) * 1.0              # N x p
    E_node[, jj] <- as.numeric(d %*% indicators) / sum_d
  }

  # Edge moments: weighted Gram of MY
  E_edge <- crossprod(MY * d, MY) / sum_d            # p x p

  list(log_Z_ratio = log_Z_ratio, E_node = E_node, E_edge = E_edge)
}

# ---------------------------------------------------------------------
# Log-likelihood gradient over the full dataset
# ---------------------------------------------------------------------
# For each individual, compute the residual between observed and expected
# sufficient statistics; then contract along the individual axis with
# X_tilde to get the gradient with respect to each covariate slope.
#
# Returns a list with grad_node (p x M x (q+1)) and grad_edge (p x p x (q+1)).

loglike_grad_individual <- function(Theta_node, Theta_edge, X_tilde, Y_obs,
                                    N = 1000) {
  n  <- nrow(Y_obs)
  p  <- ncol(Y_obs)
  M  <- dim(Theta_node)[2]
  q1 <- dim(Theta_node)[3]

  pars   <- make_individual_params(Theta_node, Theta_edge, X_tilde)
  Tn_all <- pars$Tn_all
  Te_all <- pars$Te_all

  D_node <- array(0, c(n, p, M))                     # observed - expected, per i
  D_edge <- array(0, c(n, p, p))

  for (i in 1:n) {
    Tn_i <- matrix(Tn_all[i, , ], nrow = p, ncol = M)
    Te_i <- matrix(Te_all[i, , ], nrow = p, ncol = p)

    # Observed sufficient statistics for individual i.
    S_node_i <- matrix(0, p, M)
    for (jj in 1:M) S_node_i[, jj] <- as.numeric(Y_obs[i, ] <= (jj - 1))

    my_i     <- (M - Y_obs[i, ]) / M
    S_edge_i <- tcrossprod(my_i)                     # p x p

    z <- z_moments_individual(Tn_i, Te_i, N)

    D_node[i, , ] <- S_node_i - z$E_node
    D_edge[i, , ] <- S_edge_i - z$E_edge
  }

  # Tensor contraction: grad[..., v] = sum_i X_tilde[i, v] * D[i, ...]
  # Flatten the (s, j) or (s, t) axes, do a single matrix multiply, reshape.
  D_node_mat    <- matrix(D_node, nrow = n, ncol = p * M)
  grad_node_mat <- crossprod(X_tilde, D_node_mat)    # (q+1) x (p*M)
  grad_node     <- aperm(array(grad_node_mat, c(q1, p, M)), c(2, 3, 1))

  D_edge_mat    <- matrix(D_edge, nrow = n, ncol = p * p)
  grad_edge_mat <- crossprod(X_tilde, D_edge_mat)    # (q+1) x (p*p)
  grad_edge     <- aperm(array(grad_edge_mat, c(q1, p, p)), c(2, 3, 1))

  list(grad_node = grad_node / n, grad_edge = grad_edge / n)
}

# ---------------------------------------------------------------------
# Approximate log-likelihood (for tracking convergence)
# ---------------------------------------------------------------------
# log p(Y | Theta) = sum_i [ lin_i - log Z_i ],
# where log Z_i = log Z_{0,i} + log(Z_i / Z_{0,i}) with the second term
# estimated by importance sampling.

approx_loglik_individual <- function(Theta_node, Theta_edge, X_tilde, Y_obs,
                                     N_track = 5000) {
  n <- nrow(Y_obs); p <- ncol(Y_obs); M <- dim(Theta_node)[2]

  pars   <- make_individual_params(Theta_node, Theta_edge, X_tilde)
  Tn_all <- pars$Tn_all
  Te_all <- pars$Te_all

  ll <- 0.0
  for (i in 1:n) {
    Tn_i <- matrix(Tn_all[i, , ], nrow = p, ncol = M)
    Te_i <- matrix(Te_all[i, , ], nrow = p, ncol = p)

    # Linear (sufficient-statistic) part for individual i.
    lin_node <- 0
    for (jj in 1:M) {
      lin_node <- lin_node + sum(Tn_i[, jj] * as.numeric(Y_obs[i, ] <= jj - 1))
    }
    my_i     <- (M - Y_obs[i, ]) / M
    lin_edge <- 0.5 * as.numeric(crossprod(my_i, Te_i %*% my_i))

    # log Z_i.
    log_Z0_i <- 0
    for (s in 1:p) {
      ns <- node_stat(Tn_i[s, ])
      m  <- max(ns)
      log_Z0_i <- log_Z0_i + m + log(sum(exp(ns - m)))
    }
    z <- z_moments_individual(Tn_i, Te_i, N_track)
    logZ_i <- log_Z0_i + z$log_Z_ratio

    ll <- ll + lin_node + lin_edge - logZ_i
  }
  ll
}

# ---------------------------------------------------------------------
# Proximal gradient ascent for the individual model
# ---------------------------------------------------------------------
# Mirrors proximal_grad_ascent from the base file but operates on 3D
# parameter arrays.
#
# By default the L1 penalty is applied element-wise to every entry of
# Theta_edge (intercept and all covariate slopes). To enable variable
# selection at the (s,t) edge level rather than at the (s,t,v) coefficient
# level, set penalty = "group" to soft-threshold the entire length-(q+1)
# vector of coefficients for each (s,t) edge as a group.

soft_threshold_group <- function(M_arr, tau) {
  # M_arr is p x p x (q+1); soft-threshold each length-(q+1) fibre.
  p  <- dim(M_arr)[1]
  q1 <- dim(M_arr)[3]
  fibre_norms <- sqrt(apply(M_arr^2, c(1, 2), sum))
  shrink <- pmax(1 - tau / pmax(fibre_norms, 1e-12), 0)
  for (v in 1:q1) M_arr[, , v] <- M_arr[, , v] * shrink
  M_arr
}

proximal_grad_ascent_individual <- function(Theta_node_init, Theta_edge_init,
                                            X_tilde, Y_obs,
                                            N         = 1000,
                                            step_size = 0.01,
                                            epsilon   = 1e-4,
                                            max_iter  = 500,
                                            lambda    = 0,
                                            track_obj   = FALSE,
                                            track_every = 50,
                                            N_track     = 5000,
                                            verbose   = TRUE) {
  Theta_node <- Theta_node_init
  Theta_edge <- Theta_edge_init
  p  <- dim(Theta_node)[1]
  M  <- dim(Theta_node)[2]
  q1 <- dim(Theta_node)[3]

  obj_iters  <- integer(0)
  obj_values <- numeric(0)
  if (track_obj) {
    obj_iters  <- c(obj_iters, 0)
    obj_values <- c(obj_values,
                    approx_loglik_individual(Theta_node, Theta_edge,
                                             X_tilde, Y_obs, N_track))
  }

  conv <- Inf
  iter <- 0

  while (conv > epsilon && iter < max_iter) {
    rate <- 1.0 / (1.0 + 0.01 * iter)

    g <- loglike_grad_individual(Theta_node, Theta_edge, X_tilde, Y_obs, N)
    Tn_new <- Theta_node + rate * step_size * g$grad_node
    Te_new <- Theta_edge + rate * step_size * g$grad_edge

    # Symmetrize every slice and zero diagonals.
    for (v in 1:q1) {
      Te_new[, , v] <- 0.5 * (Te_new[, , v] + t(Te_new[, , v]))
      diag(Te_new[, , v]) <- 0
    }

    # Proximal step (group lasso).
    if (lambda > 0) {
      tau <- rate * step_size * lambda
      Te_new <- soft_threshold_group(Te_new, tau)
      for (v in 1:q1) diag(Te_new[, , v]) <- 0
    }

    nrm  <- sum(Theta_node^2) + sum(Theta_edge^2)
    diff <- sum((Tn_new - Theta_node)^2) + sum((Te_new - Theta_edge)^2)
    conv <- sqrt(diff / max(nrm, 1e-12))

    Theta_node <- Tn_new
    Theta_edge <- Te_new
    iter <- iter + 1

    if (track_obj && (iter %% track_every == 0)) {
      ll <- approx_loglik_individual(Theta_node, Theta_edge,
                                     X_tilde, Y_obs, N_track)
      obj_iters  <- c(obj_iters, iter)
      obj_values <- c(obj_values, ll)
      if (verbose) {
        cat(sprintf("iter = %d, conv = %.6f, approx loglik = %.4f\n",
                    iter, conv, ll))
      }
    } else if (verbose && (iter %% 50 == 0)) {
      cat(sprintf("iter = %d, convergence = %.6f\n", iter, conv))
    }
  }

  if (verbose) {
    cat(sprintf("Stopped at iter %d, convergence = %.6f\n", iter, conv))
  }

  out <- list(Theta_node = Theta_node,
              Theta_edge = Theta_edge,
              iter       = iter,
              conv       = conv,
              step_size  = step_size,
              N          = N,
              lambda     = lambda,
              epsilon    = epsilon)
  if (track_obj) {
    out$obj_iters  <- obj_iters
    out$obj_values <- obj_values
  }
  out
}

# ---------------------------------------------------------------------
# Pseudo-likelihood via ordinalNet (covariate-dependent edges)
# ---------------------------------------------------------------------
# For each node s we fit the same adjacent-category (acat) pseudo-conditional
# as in fit_ordinalNet(), but the linear predictor for neighbours includes both
#   (M - Y_{it})                     -- baseline edge (s, t)
#   (M - Y_{it}) * X_{ik}             -- how covariate k modulates that edge
# for every t != s.  With parallelTerms = TRUE these effects are shared across
# ordinal categories, which matches a pseudo-conditional of the form where an
# effective coupling theta_st(x_i) is linear in x_tilde_i and multiplies the
# neighbour score M - Y_{it} (same scaling convention as fit_ordinalNet).
#
# Primary output: Theta_edge as a p x p x (q+1) array, symmetrized slice-wise,
# then scaled by M^2 like fit_ordinalNet() so it matches the rest of this file.
#
# The ordinalNet *intercepts* are returned as acat_intercepts (p x M): they are
# baseline adjacent-category parameters for each pseudo-conditional and do *not*
# coincide with the joint CR-MRF node potentials Theta_node.  For likelihood-
# based fitting of Theta_node, use proximal_grad_ascent_individual or another
# method; here we focus on a usable Theta_edge initialiser / comparator.

fit_ordinalNet_individual <- function(Y_obs, X, M,
                                    symmetrize = c("avg", "and", "none"),
                                    alpha = 1, standardize = FALSE,
                                    criteria = c("aic", "bic"),
                                    which_lambda = NULL, ...) {
  if (!requireNamespace("ordinalNet", quietly = TRUE)) {
    stop("Package 'ordinalNet' is required for fit_ordinalNet_individual().")
  }
  if (!requireNamespace("foreach", quietly = TRUE)) {
    stop("Package 'foreach' is required for fit_ordinalNet_individual().")
  }

  symmetrize <- match.arg(symmetrize)
  criteria   <- match.arg(criteria)

  n <- nrow(Y_obs)
  p <- ncol(Y_obs)
  X <- if (is.null(X)) {
    matrix(0, nrow = n, ncol = 0)
  } else {
    as.matrix(X)
  }
  q <- ncol(X)
  if (q > 0L && nrow(X) != n) {
    stop("nrow(X) must equal nrow(Y_obs) when covariates are supplied.")
  }
  q1 <- q + 1L

  Theta_edge <- array(0, dim = c(p, p, q1))
  acat_intercepts <- matrix(0, nrow = p, ncol = M)
  intercept_slopes<- matrix(0, nrow = p, ncol = q)
  fit_list       <- vector("list", p)
  lambda_index   <- integer(p)
  lambda_vals    <- numeric(p)

  U <- M - Y_obs

  # Capture `...` explicitly: foreach does not reliably forward `...` into
  # %dopar% workers, so we splice it back in via do.call() inside the body.
  dots <- list(...)

  # Fit each node's pseudo-conditional independently (embarrassingly parallel).
  # Each iteration returns a self-contained list; results are stitched into the
  # shared containers afterward so workers never write to disjoint slices.
  s <- NULL  # silence R CMD check note for the foreach iterator
  node_results <- foreach::foreach(s = seq_len(p)) %dopar% {
    y_s <- ordered(Y_obs[, s], levels = 0:M)
    others <- (seq_len(p))[-s]

    if (q == 0L) {
      x_s <- as.matrix(U[, others, drop = FALSE])
      colnames(x_s) <- paste0("V", others)
    } else {
      blk <- list()
      cn  <- character((p - 1L) * q1)
      z <- 1L
      for (t in others) {
        vt <- U[, t]
        cn[z] <- sprintf("V%d_base", t)
        blk[[length(blk) + 1L]] <- vt
        z <- z + 1L
        for (k in seq_len(q)) {
          cn[z] <- sprintf("V%d_x%d", t, k)
          blk[[length(blk) + 1L]] <- vt * X[, k]
          z <- z + 1L
        }
      }
      x_s <- do.call(cbind, blk)
      colnames(x_s) <- cn
      X_named <- X
      colnames(X_named) <- paste0("x_", seq_len(q))
      x_s <- cbind(X_named, x_s)
    }

    # Penalty factors over the columns of x_s.  The node's own covariate
    # slopes (the leading q columns, named x_1..x_q) are EXEMPTED (factor 0)
    # so only the edge terms (V*) are penalized -- matching the joint MLE,
    # which penalizes edges only.  Category intercepts are never penalized by
    # ordinalNet regardless.
    pf <- rep(1, ncol(x_s))
    if (q > 0L) pf[seq_len(q)] <- 0

    fit_s <- do.call(ordinalNet::ordinalNet, c(list(
      x = x_s,
      y = y_s,
      family = "acat",
      reverse = TRUE,
      parallelTerms = TRUE,
      nonparallelTerms = FALSE,
      alpha = alpha,
      standardize = standardize,
      penaltyFactors = pf
    ), dots))

    idx <- which_lambda
    if (is.null(idx)) idx <- which.min(fit_s[[criteria]])

    cc <- stats::coef(fit_s, whichLambda = idx, criteria = criteria)

    intercept_names <- grep("^\\(Intercept\\):", names(cc), value = TRUE)
    acat_row <- unname(cc[intercept_names])

    intercept_slope_names <- grep("^x_", names(cc), value = TRUE)
    islope_row <- unname(cc[intercept_slope_names])

    slope_names <- grep("^V", names(cc), value = TRUE)
    slopes <- unname(cc[slope_names])

    # Map slopes -> directed Theta_edge[s, t, ] (filled from regression at s).
    edge_row <- matrix(0, nrow = p, ncol = q1)
    for (j in seq_along(others)) {
      t <- others[j]
      base_c <- (j - 1L) * q1 + 1L
      edge_row[t, 1L] <- slopes[base_c]
      if (q > 0L) {
        for (k in seq_len(q)) {
          edge_row[t, k + 1L] <- slopes[base_c + k]
        }
      }
    }

    list(acat_row = acat_row, islope_row = islope_row, edge_row = edge_row,
         fit = fit_s, idx = idx, lambda = fit_s$lambdaVals[idx])
  }

  # Stitch per-node results into the shared containers.
  for (s in seq_len(p)) {
    res <- node_results[[s]]
    acat_intercepts[s, ] <- res$acat_row
    if (q > 0L) intercept_slopes[s, ] <- res$islope_row
    Theta_edge[s, , ]    <- res$edge_row
    fit_list[[s]]        <- res$fit
    lambda_index[s]      <- res$idx
    lambda_vals[s]       <- res$lambda
  }

  for (v in seq_len(q1)) {
    Theta_edge[, , v] <- .symmetrize_theta_edge(
      Theta_edge[, , v], symmetrize = symmetrize
    )
  }
  Theta_edge <- Theta_edge * (M^2)

  list(
    Theta_edge       = Theta_edge,
    acat_intercepts  = acat_intercepts,
    intercept_slopes = intercept_slopes,
    fit              = fit_list,
    lambda_index     = lambda_index,
    lambda           = lambda_vals,
    M                = M,
    q                = q,
    symmetrize       = symmetrize
  )
}

# ---------------------------------------------------------------------
# Pseudo-likelihood via VGAM (covariate-dependent edges, unpenalised)
# ---------------------------------------------------------------------
# Same per-node adjacent-category pseudo-conditional as
# fit_ordinalNet_individual(), but fit by maximum likelihood via
# VGAM::vglm with family = acat(reverse = TRUE, parallel = TRUE).
# VGAM has no built-in elastic-net penalty, so there is no alpha /
# standardize / criteria / which_lambda / lambda_index machinery.

fit_vgam_individual <- function(Y_obs, X, M,
                                symmetrize = c("avg", "and", "none"),
                                ...) {
  if (!requireNamespace("VGAM", quietly = TRUE)) {
    stop("Package 'VGAM' is required for fit_vgam_individual().")
  }

  symmetrize <- match.arg(symmetrize)

  n <- nrow(Y_obs)
  p <- ncol(Y_obs)
  X <- if (is.null(X)) {
    matrix(0, nrow = n, ncol = 0)
  } else {
    as.matrix(X)
  }
  q <- ncol(X)
  if (q > 0L && nrow(X) != n) {
    stop("nrow(X) must equal nrow(Y_obs) when covariates are supplied.")
  }
  q1 <- q + 1L

  Theta_edge       <- array(0, dim = c(p, p, q1))
  acat_intercepts  <- array(0, dim = c(p, M, q1))
  fit_list         <- vector("list", p)

  U <- M - Y_obs

  for (s in seq_len(p)) {
    cat(sprintf("  [VGAM individual] node %d / %d...\n", s, p))

    y_s    <- ordered(Y_obs[, s], levels = 0:M)
    others <- (seq_len(p))[-s]

    if (q == 0L) {
      x_s <- as.matrix(U[, others, drop = FALSE])
      colnames(x_s) <- paste0("V", others)
    } else {
      blk <- list()
      cn  <- character((p - 1L) * q1)
      z <- 1L
      for (t in others) {
        vt <- U[, t]
        cn[z] <- sprintf("V%d_base", t)
        blk[[length(blk) + 1L]] <- vt
        z <- z + 1L
        for (k in seq_len(q)) {
          cn[z] <- sprintf("V%d_x%d", t, k)
          blk[[length(blk) + 1L]] <- vt * X[, k]
          z <- z + 1L
        }
      }
      x_s <- do.call(cbind, blk)
      colnames(x_s) <- cn
      X_named <- X
      colnames(X_named) <- paste0("x_", seq_len(q))
      x_s <- cbind(X_named, x_s)
    }

    df_s   <- data.frame(.y = y_s, x_s, check.names = FALSE)
    form_s <- stats::as.formula(
      paste(".y ~", paste(colnames(x_s), collapse = " + "))
    )

    nonpar_terms <- grep("^x_", colnames(x_s), value = TRUE)
    par_arg <- if (length(nonpar_terms) > 0L) {
      stats::as.formula(
        paste("FALSE ~", paste(nonpar_terms, collapse = " + "))
      )
    } else {
      TRUE
    }

    fit_s <- VGAM::vglm(
      form_s,
      family = VGAM::acat(reverse = TRUE, parallel = par_arg),
      data   = df_s,
      ...
    )

    cc <- VGAM::coef(fit_s)

    intercept_names <- grep("^\\(Intercept\\):", names(cc), value = TRUE)
    acat_intercepts[s, , 1L] <- unname(cc[intercept_names])

    if (q > 0L) {
      for (qi in seq_len(q)) {
        q_pat <- paste0("^x_", qi, ":")
        intercept_slope_names <- grep(q_pat, names(cc), value = TRUE)
        acat_intercepts[s, , qi + 1L] <- unname(cc[intercept_slope_names])
      }
    }

    for (j in seq_along(others)) {
      t <- others[j]
      if (q == 0L) {
        Theta_edge[s, t, 1L] <- unname(cc[paste0("V", t)])
      } else {
        Theta_edge[s, t, 1L] <- unname(cc[sprintf("V%d_base", t)])
        for (k in seq_len(q)) {
          Theta_edge[s, t, k + 1L] <- unname(cc[sprintf("V%d_x%d", t, k)])
        }
      }
    }

    fit_list[[s]] <- fit_s
  }

  for (v in seq_len(q1)) {
    Theta_edge[, , v] <- .symmetrize_theta_edge(
      Theta_edge[, , v], symmetrize = symmetrize
    )
  }
  Theta_edge <- Theta_edge * (M^2)

  list(
    Theta_edge       = Theta_edge,
    acat_intercepts  = acat_intercepts,
    fit              = fit_list,
    M                = M,
    q                = q,
    symmetrize       = symmetrize
  )
}

# ---------------------------------------------------------------------
# Helper: approximate lambda_max for one node in the individual model
# ---------------------------------------------------------------------
# Extends lambda_max_node() from the base file to the individual model's
# extended feature matrix, which includes both the baseline neighbour
# scores U[,t] and their interactions U[,t]*X[,k] for k = 1,...,q.
# Uses the same linear approximation: max |x_s^T r| / n where r is the
# mean-centred response U[,s].

lambda_max_node_individual <- function(s, Y_obs, X, M) {
  n      <- nrow(Y_obs)
  U      <- M - Y_obs                          # n x p
  u_s    <- U[, s]
  r      <- u_s - mean(u_s)                    # centred response
  others <- seq_len(ncol(Y_obs))[-s]
  q      <- if (is.null(X)) 0L else ncol(X)

  if (q == 0L) {
    x_s <- U[, others, drop = FALSE]
  } else {
    blk <- vector("list", length(others) * (q + 1L))
    z   <- 1L
    for (t in others) {
      vt       <- U[, t]
      blk[[z]] <- vt
      z        <- z + 1L
      for (k in seq_len(q)) {
        blk[[z]] <- vt * X[, k]
        z        <- z + 1L
      }
    }
    x_s <- do.call(cbind, blk)
  }

  max(abs(crossprod(x_s, r))) / n
}

# ---------------------------------------------------------------------
# CV-tuned pseudo-likelihood with a shared lambda path (individual model)
# ---------------------------------------------------------------------
# Extends fit_ordinalNet_cv_shared_lambda() to the individual CR-MRF,
# where the design matrix for each node s includes both baseline neighbour
# scores and their interactions with covariates X.
#
# Arguments mirror fit_ordinalNet_individual() plus the CV arguments from
# fit_ordinalNet_cv_shared_lambda():
#   Y_obs       : n x p integer matrix of observed counts (0..M)
#   X           : n x q covariate matrix (NULL or zero-column if q = 0)
#   M           : ordinal upper bound
#   symmetrize  : how to symmetrize the directed Theta_edge slices
#   tuneMethod  : CV criterion passed to ordinalNetTune
#   lambdaVals  : optional fixed lambda path (computed from data if NULL)
#   nFolds      : number of CV folds
#   which_lambda: integer index into the lambda path to select; if NULL,
#                 chosen by the CV criterion
#   printProgress: print per-node progress messages
#   lambdaMinRatio, nLambda: control the auto-generated lambda path
#
# Returns a list mirroring fit_ordinalNet_individual() plus tuning info:
#   Theta_edge, acat_intercepts, tunefit, lambda_index, lambda,
#   lambda_shared, lambdaVals, score_path, aggregated_score, metric

fit_ordinalNet_cv_shared_lambda_individual <- function(
    Y_obs, X, M,
    symmetrize     = c("avg", "and", "none"),
    alpha          = 1,
    standardize    = FALSE,
    tuneMethod     = c("cvLoglik", "cvMisclass", "cvBrier", "cvDevPct"),
    lambdaVals     = NULL,
    nFolds         = 5,
    which_lambda   = NULL,
    printProgress  = TRUE,
    lambdaMinRatio = 0.01,
    nLambda        = 10,
    ...) {

  if (!requireNamespace("ordinalNet", quietly = TRUE)) {
    stop("Package 'ordinalNet' is required for ",
         "fit_ordinalNet_cv_shared_lambda_individual().")
  }
  if (!requireNamespace("foreach", quietly = TRUE)) {
    stop("Package 'foreach' is required for ",
         "fit_ordinalNet_cv_shared_lambda_individual().")
  }

  symmetrize <- match.arg(symmetrize)
  tuneMethod <- match.arg(tuneMethod)

  n <- nrow(Y_obs)
  p <- ncol(Y_obs)
  X <- if (is.null(X)) matrix(0, nrow = n, ncol = 0L) else as.matrix(X)
  q <- ncol(X)
  if (q > 0L && nrow(X) != n) {
    stop("nrow(X) must equal nrow(Y_obs) when covariates are supplied.")
  }
  q1 <- q + 1L

  # Build shared lambda path from data if not supplied.
  if (is.null(lambdaVals)) {
    X_for_lmax   <- if (q == 0L) NULL else X
    # Wrap in a closure: passing `X = ...` through vapply(...) would collide
    # with vapply's own first formal argument `X`.
    lambda_maxes <- vapply(
      seq_len(p),
      function(s) lambda_max_node_individual(s, Y_obs = Y_obs,
                                             X = X_for_lmax, M = M),
      numeric(1L)
    )
    lambda_max_shared <- max(lambda_maxes)
    lambdaVals <- exp(seq(log(lambda_max_shared),
                          log(lambda_max_shared * lambdaMinRatio),
                          length.out = nLambda))
  }

  Theta_edge       <- array(0, dim = c(p, p, q1))
  acat_intercepts  <- matrix(0, nrow = p, ncol = M)
  intercept_slopes <- matrix(0, nrow = p, ncol = q)

  tunefit_list    <- vector("list", p)
  score_path      <- matrix(NA_real_, nrow = length(lambdaVals), ncol = p)
  colnames(score_path) <- paste0("node", seq_len(p))

  metric_name <- switch(
    tuneMethod,
    cvLoglik  = "loglik",
    cvMisclass = "misclass",
    cvBrier   = "brier",
    cvDevPct  = "devPct"
  )

  U <- M - Y_obs                              # n x p

  # Capture `...` explicitly: foreach does not reliably forward `...` into
  # %dopar% workers, so we splice it back in via do.call() inside the body.
  dots <- list(...)

  # Tune each node over the shared lambda path independently (parallel over s).
  s <- NULL  # silence R CMD check note for the foreach iterator
  node_tunes <- foreach::foreach(s = seq_len(p)) %dopar% {
    y_s    <- ordered(Y_obs[, s], levels = 0:M)
    others <- seq_len(p)[-s]

    # Build the same extended design matrix as fit_ordinalNet_individual().
    if (q == 0L) {
      x_s <- as.matrix(U[, others, drop = FALSE])
      colnames(x_s) <- paste0("V", others)
    } else {
      blk <- list()
      cn  <- character(length(others) * q1)
      z   <- 1L
      for (t in others) {
        vt          <- U[, t]
        cn[z]       <- sprintf("V%d_base", t)
        blk[[z]]    <- vt
        z           <- z + 1L
        for (k in seq_len(q)) {
          cn[z]    <- sprintf("V%d_x%d", t, k)
          blk[[z]] <- vt * X[, k]
          z        <- z + 1L
        }
      }
      x_s <- do.call(cbind, blk)
      colnames(x_s) <- cn
      X_named <- X
      colnames(X_named) <- paste0("x_", seq_len(q))
      x_s <- cbind(X_named, x_s)
    }

    # Penalty factors over the columns of x_s.  The node's own covariate
    # slopes (the leading q columns, named x_1..x_q) are EXEMPTED (factor 0)
    # so only the edge terms (V*) are penalized -- matching
    # fit_ordinalNet_individual() and lambda_max_node_individual(), which
    # both treat edges as the only penalized terms.
    pf <- rep(1, ncol(x_s))
    if (q > 0L) pf[seq_len(q)] <- 0

    tunefit_s <- do.call(ordinalNet::ordinalNetTune, c(list(
      x             = x_s,
      y             = y_s,
      lambdaVals    = lambdaVals,
      nFolds        = nFolds,
      family        = "acat",
      reverse       = TRUE,
      parallelTerms = TRUE,
      nonparallelTerms = FALSE,
      alpha         = alpha,
      standardize   = standardize,
      penaltyFactors = pf,
      printProgress = FALSE
    ), dots))

    list(score_col = rowMeans(tunefit_s[[metric_name]], na.rm = TRUE),
         tunefit   = tunefit_s)
  }

  for (s in seq_len(p)) {
    score_path[, s]   <- node_tunes[[s]]$score_col
    tunefit_list[[s]] <- node_tunes[[s]]$tunefit
  }

  # Aggregate CV scores across nodes and pick lambda.
  agg_score <- rowMeans(score_path, na.rm = TRUE)
  idx <- which_lambda
  if (is.null(idx)) {
    if (tuneMethod %in% c("cvLoglik", "cvDevPct")) {
      idx <- which.max(agg_score)
    } else {
      idx <- which.min(agg_score)
    }
  }

  # Extract coefficients at the selected lambda into Theta_edge (3D) and
  # acat_intercepts.
  for (s in seq_len(p)) {
    others <- seq_len(p)[-s]
    cc     <- stats::coef(tunefit_list[[s]]$fit, whichLambda = idx)

    intercept_names     <- grep("^\\(Intercept\\):", names(cc), value = TRUE)
    acat_intercepts[s, ] <- unname(cc[intercept_names])

    if (q > 0L) {
      intercept_slope_names <- grep("^x_", names(cc), value = TRUE)
      intercept_slopes[s, ] <- unname(cc[intercept_slope_names])
    }

    # Map slopes back to Theta_edge[s, t, v].
    for (j in seq_along(others)) {
      t <- others[j]

      if (q == 0L) {
        slope_name <- paste0("V", t)
        Theta_edge[s, t, 1L] <- unname(cc[slope_name])
      } else {
        Theta_edge[s, t, 1L] <- unname(cc[sprintf("V%d_base", t)])
        for (k in seq_len(q)) {
          Theta_edge[s, t, k + 1L] <- unname(cc[sprintf("V%d_x%d", t, k)])
        }
      }
    }
  }

  # Symmetrize each slice and scale.
  for (v in seq_len(q1)) {
    Theta_edge[, , v] <- .symmetrize_theta_edge(
      Theta_edge[, , v], symmetrize = symmetrize
    )
  }
  Theta_edge <- Theta_edge * (M^2)

  list(
    Theta_edge        = Theta_edge,
    acat_intercepts   = acat_intercepts,
    intercept_slopes  = intercept_slopes,
    tunefit           = tunefit_list,
    lambda_index      = rep(idx, p),
    lambda            = rep(lambdaVals[idx], p),
    lambda_shared     = lambdaVals[idx],
    lambdaVals        = lambdaVals,
    score_path        = score_path,
    aggregated_score  = agg_score,
    metric            = metric_name,
    M                 = M,
    q                 = q,
    symmetrize        = symmetrize
  )
}


# # ---------------------------------------------------------------------
# # Demo (commented out)
# # ---------------------------------------------------------------------

# p <- 3 ; M <- 5; q <- 1; n <- 300

# # True parameters
# Theta_node_true <- array(runif(p * M * (q + 1), -0.5, 0.5), c(p, M, q + 1))

# Theta_edge_true <- array(0, c(p, p, q + 1))
# # Intercept slice: a chain
# for (s in 1:(p - 1)) Theta_edge_true[s, s + 1, 1] <- -1
# Theta_edge_true[, , 1] <- Theta_edge_true[, , 1] + t(Theta_edge_true[, , 1])
# # Covariate slices: a few sparse couplings
# for (v in 2:(q + 1)) {
#   E <- matrix(0, p, p)
#   ut_idx <- which(upper.tri(E))
#   E[sample(ut_idx, 2)] <- -1
#   E <- E + t(E)
#   Theta_edge_true[, , v] <- E
# }

# # Covariates and augmented matrix
# X       <- matrix(0, n, q)
# X_tilde <- augment_X(X)

# # Simulate
# Y_obs <- gibbs_cr_mrf_individual(Theta_node_true, Theta_edge_true,
#                                  X_tilde, n_iter = 300)

# # Estimate
# Theta_node_init <- array(0, c(p, M, q + 1))
# Theta_edge_init <- array(0, c(p, p, q + 1))

# fit <- proximal_grad_ascent_individual(
#   Theta_node_init, Theta_edge_init,
#   X_tilde, Y_obs,
#   #penalty = "group",
#   N         = 1000,
#   step_size = 0.05,
#   max_iter  = 2000,
#   lambda    = 0,
#   track_obj   = TRUE,
#   track_every = 100,
#   N_track     = 2000
# )

# # =====================================================================
# # Sanity check: with no covariates (q = 0), the individual model must
# # reduce, in distribution, to the base count_graphical model.
# #
# # We compare three things between the two implementations:
# #   (1) marginal and pairwise frequencies of simulated Y
# #   (2) one log-likelihood gradient at fixed Theta (large N)
# #   (3) the final fit returned by proximal gradient ascent
# #
# # Caveats:
# #   - Independence sampling and importance sampling use different RNG
# #     streams and different IS strategies (one shared batch vs n
# #     per-individual batches), so numerical agreement is up to MC error.
# # =====================================================================

# cat("\n\n############################################################\n")
# cat("# Sanity check: q = 0 should match base count_graphical model #\n")
# cat("############################################################\n")

# set.seed(2024)

# p_sc <- 3
# M_sc <- 5
# n_sc <- 500

# # Base-model true parameters.
# Theta_node_base <- matrix(runif(p_sc * M_sc, -0.5, 0.5),
#                           nrow = p_sc, ncol = M_sc)
# Theta_edge_base <- matrix(0, nrow = p_sc, ncol = p_sc)
# for (s in 1:(p_sc - 1)) Theta_edge_base[s, s + 1] <- - 0.4
# Theta_edge_base <- Theta_edge_base + t(Theta_edge_base)

# # Lift to individual-model parameters with q = 0 (only the intercept slice).
# Theta_node_indiv <- array(Theta_node_base, c(p_sc, M_sc, 1))
# Theta_edge_indiv <- array(Theta_edge_base, c(p_sc, p_sc, 1))

# # X has zero columns; X_tilde is just an n x 1 vector of 1s.
# X_tilde_empty <- matrix(1, n_sc, 1)

# # ---------------------------------------------------------------------
# # (1) Compare simulated marginal and pairwise frequencies.
# # ---------------------------------------------------------------------
# cat("\n--- (1) Empirical distribution of Y ---\n")

# set.seed(1)
# Y_base <- gibbs_cr_mrf(Theta_node_base, Theta_edge_base,
#                        n = n_sc, n_iter = 500)
# set.seed(2)
# Y_indiv <- gibbs_cr_mrf_individual(Theta_node_indiv, Theta_edge_indiv,
#                                    X_tilde_empty, n_iter = 500)

# cat("Marginal frequencies (rows = node, columns = 0..M):\n")
# for (s in 1:p_sc) {
#   tab_b <- prop.table(tabulate(Y_base[, s]  + 1L, nbins = M_sc + 1))
#   tab_i <- prop.table(tabulate(Y_indiv[, s] + 1L, nbins = M_sc + 1))
#   cat(sprintf("  node %d  base : %s\n", s,
#               paste(sprintf("%.3f", tab_b), collapse = " ")))
#   cat(sprintf("          indv : %s    max|diff| = %.3f\n",
#               paste(sprintf("%.3f", tab_i), collapse = " "),
#               max(abs(tab_b - tab_i))))
# }

# cat("\nPairwise joint frequencies, max |diff| over the (M+1)^2 cells:\n")
# for (s in 1:(p_sc - 1)) {
#   for (t in (s + 1):p_sc) {
#     pb <- prop.table(table(factor(Y_base[, s],  levels = 0:M_sc),
#                            factor(Y_base[, t],  levels = 0:M_sc)))
#     pi <- prop.table(table(factor(Y_indiv[, s], levels = 0:M_sc),
#                            factor(Y_indiv[, t], levels = 0:M_sc)))
#     cat(sprintf("  (%d,%d): max |diff| = %.3f\n",
#                 s, t, max(abs(pb - pi))))
#   }
# }

# # ---------------------------------------------------------------------
# # (2) Compare a single log-likelihood gradient evaluation.
# # ---------------------------------------------------------------------
# cat("\n--- (2) Gradient at the true Theta (large N) ---\n")

# N_grad <- 20000

# set.seed(11)
# g_base  <- loglike_grad(Theta_node_base, Theta_edge_base,
#                         Y_base, N = N_grad)
# set.seed(11)
# g_indiv <- loglike_grad_individual(Theta_node_indiv, Theta_edge_indiv,
#                                    X_tilde_empty, Y_base, N = N_grad)

# # Relative discrepancy, normalised by the base gradient magnitude.
# node_err <- max(abs(g_base$grad_node - g_indiv$grad_node[, , 1]))
# edge_err <- max(abs(g_base$grad_edge - g_indiv$grad_edge[, , 1]))
# cat(sprintf("  max |grad_node base - indiv[..,1]| = %.4f  (base ||.||_inf = %.4f)\n",
#             node_err, max(abs(g_base$grad_node))))
# cat(sprintf("  max |grad_edge base - indiv[..,1]| = %.4f  (base ||.||_inf = %.4f)\n",
#             edge_err, max(abs(g_base$grad_edge))))

# # ---------------------------------------------------------------------
# # (3) Compare the final fit from proximal gradient ascent.
# # ---------------------------------------------------------------------
# cat("\n--- (3) Final proximal-gradient-ascent fit ---\n")

# Tn_init_base <- matrix(0, p_sc, M_sc)
# Te_init_base <- matrix(0, p_sc, p_sc)
# Tn_init_indv <- array(0, c(p_sc, M_sc, 1))
# Te_init_indv <- array(0, c(p_sc, p_sc, 1))

# cat("Fitting base model...\n")
# set.seed(101)
# fit_base <- proximal_grad_ascent(
#   Tn_init_base, Te_init_base,
#   Y_obs     = Y_base,
#   N         = 2000,
#   step_size = 0.05,
#   epsilon   = 1e-4,
#   max_iter  = 1000,
#   lambda    = 0,
#   verbose   = FALSE
# )

# cat("Fitting individual model with q = 0...\n")
# set.seed(101)
# fit_indv <- proximal_grad_ascent_individual(
#   Tn_init_indv, Te_init_indv,
#   X_tilde   = X_tilde_empty,
#   Y_obs     = Y_base,
#   N         = 2000,
#   step_size = 0.05,
#   epsilon   = 1e-4,
#   max_iter  = 1000,
#   lambda    = 0,
#   verbose   = FALSE
# )

# cat(sprintf("  base : iter = %d, conv = %.5f\n",
#             fit_base$iter, fit_base$conv))
# cat(sprintf("  indv : iter = %d, conv = %.5f\n",
#             fit_indv$iter, fit_indv$conv))

# cat("\nTheta_node (true vs base vs indv[,,1]):\n")
# for (s in 1:p_sc) {
#   cat(sprintf("  s=%d  true: %s\n", s,
#               paste(sprintf("%6.3f", Theta_node_base[s, ]), collapse = " ")))
#   cat(sprintf("        base: %s\n",
#               paste(sprintf("%6.3f", fit_base$Theta_node[s, ]), collapse = " ")))
#   cat(sprintf("        indv: %s\n",
#               paste(sprintf("%6.3f", fit_indv$Theta_node[s, , 1]), collapse = " ")))
# }

# cat("\nTheta_edge upper-tri (true / base / indv):\n")
# for (s in 1:(p_sc - 1)) {
#   for (t in (s + 1):p_sc) {
#     cat(sprintf("  (%d,%d): true = %6.3f   base = %6.3f   indv = %6.3f\n",
#                 s, t,
#                 Theta_edge_base[s, t],
#                 fit_base$Theta_edge[s, t],
#                 fit_indv$Theta_edge[s, t, 1]))
#   }
# }

# cat("\nFrobenius distance between the two fits:\n")
# cat(sprintf("  ||Theta_node_base - Theta_node_indv[,,1]||_F = %.4f\n",
#             sqrt(sum((fit_base$Theta_node - fit_indv$Theta_node[, , 1])^2))))
# cat(sprintf("  ||Theta_edge_base - Theta_edge_indv[,,1]||_F = %.4f\n",
#             sqrt(sum((fit_base$Theta_edge - fit_indv$Theta_edge[, , 1])^2))))

# cat("\nFrobenius distance to the truth:\n")
# cat(sprintf("  base : node = %.4f, edge = %.4f\n",
#             sqrt(sum((fit_base$Theta_node - Theta_node_base)^2)),
#             sqrt(sum((fit_base$Theta_edge - Theta_edge_base)^2))))
# cat(sprintf("  indv : node = %.4f, edge = %.4f\n",
#             sqrt(sum((fit_indv$Theta_node[, , 1] - Theta_node_base)^2)),
#             sqrt(sum((fit_indv$Theta_edge[, , 1] - Theta_edge_base)^2))))
