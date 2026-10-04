node_stat <- function(theta_s) {
  rev(c(0, cumsum(rev(theta_s))))
}

node_pmf_indep <- function(theta_s){
  log_unnorm <- node_stat(theta_s)
  log_unnorm <- log_unnorm - max(log_unnorm) 
  unnorm <- exp(log_unnorm)
  unnorm / sum(unnorm)
}

full_conditional_pmf <- function(theta_s, r_s) {
  M <- length(theta_s)
  ys <- 0:M
  log_unnorm <- node_stat(theta_s) + (M - ys) / M * r_s
  log_unnorm <- log_unnorm - max(log_unnorm)
  unnorm <- exp(log_unnorm)
  unnorm / sum(unnorm)
}

sample_ordinal <- function(pmf) {
  sample(x = 0:(length(pmf) - 1), size = 1, prob = pmf)
}

sample_independent <- function(Theta_node, n) {
  p <- nrow(Theta_node)
  M <- ncol(Theta_node)
  X <- matrix(NA_integer_, nrow = n, ncol = p)
  for (s in 1:p) {
    pmf <- node_pmf_indep(Theta_node[s, ])
    X[, s] <- sample(0:M, size = n, replace = TRUE, prob = pmf)
  }
  X
}

gibbs_cr_mrf <- function(Theta_node, Theta_edge, n, n_iter = 500) {
  p <- nrow(Theta_node)
  M <- ncol(Theta_node)
 
  X <- sample_independent(Theta_node, n)
 
  for (iter in 1:n_iter) {
    for (s in 1:p) {
      # r_s for all n chains: n-vector
      # r_s[i] = sum_t theta_{st} * (M - X[i, t])
      r_s <- ((M - X) / M) %*% Theta_edge[s, ]   # n x 1

      # Log-unnormalized PMF: n x (M+1) matrix
      ns <- node_stat(Theta_node[s, ])                   # length M+1
      ys <- 0:M                                          # length M+1
      log_unnorm <- outer(as.numeric(r_s), (M - ys) / M) +
                    matrix(ns, nrow = n, ncol = M + 1, byrow = TRUE)
 
      # Stabilise and exponentiate
      log_unnorm <- log_unnorm - apply(log_unnorm, 1, max)
      probs <- exp(log_unnorm)
      probs <- probs / rowSums(probs)                    # n x (M+1)
 
      # Sample from each row's categorical distribution
      cdf <- t(apply(probs, 1, cumsum))                  # n x (M+1)
      u <- runif(n)
      X[, s] <- rowSums(cdf < u) # number of CDF values < u = sampled category
    }
  }
 
  X
}

log_unnorm_joint <- function(y, Theta_node, Theta_edge) {
  p <- nrow(Theta_node)
  M <- ncol(Theta_node)
  node_val <- 0
  for (s in 1:p) {
    ns <- node_stat(Theta_node[s, ])
    node_val <- node_val + ns[y[s] + 1]
  }
  edge_val <- 0
  for (s in 1:(p - 1)) {
    for (t in (s + 1):p) {
      if (Theta_edge[s, t] != 0) {
        edge_val <- edge_val + Theta_edge[s, t] * (M - y[s]) / M * (M - y[t]) / M
      }
    }
  }
  node_val + edge_val
}


z_functions <- function(Theta_node, Theta_edge, N) {
  p <- nrow(Theta_node)
  M <- ncol(Theta_node)
 
  # --- 1. Sample Y from the independence model ---
  Y <- sample_independent(Theta_node, N)   # N x p
 
  # --- 2. Compute importance weights: w_b = exp(g(Y_b)) ---
  # g(Y) = sum_{(s,t)} theta_{st} (M - Y_s)(M - Y_t)
  
  MY <- (M - Y) / M                                 # N x p, normalized to [0,1]
  q_vals <- rowSums((MY %*% Theta_edge) * MY) / 2   # N-vector
 
  # Stabilise: subtract max before exp
  q_max <- max(q_vals)
  d <- exp(q_vals - q_max)                       
  Z_ratio <- exp(q_max) * mean(d)
 
  sum_d <- sum(d)
 
  # --- 3. Weighted sufficient statistics for grad log Z ---
 
  # (a) Node gradient:
  grad_logZ_node <- matrix(0, nrow = p, ncol = M)
  for (jj in 1:M) {
    indicators <- (Y <= (jj - 1)) * 1.0           # N x p 
    grad_logZ_node[, jj] <- (d %*% indicators) / sum_d
  }
 
  # (b) Edge gradient: 
  dMY <- MY * d                                    # N x p, each row weighted
  grad_logZ_edge <- (t(dMY) %*% MY) / sum_d        # p x p
 
  list(
    Z_ratio        = Z_ratio,
    grad_logZ_node = grad_logZ_node,
    grad_logZ_edge = grad_logZ_edge
  )
}

loglike_grad <- function(Theta_node, Theta_edge, Y_obs, N) {
  n <- nrow(Y_obs)
  p <- ncol(Y_obs)
  M <- ncol(Theta_node)
 

  S_node <- matrix(0, nrow = p, ncol = M)
  for (jj in 1:M) {
    S_node[, jj] <- colSums(Y_obs <= (jj - 1))
  }
 
  MY_obs <- (M - Y_obs) / M                         # n x p, normalized to [0,1]
  S_edge <- (t(MY_obs) %*% MY_obs)                 # p x p
 
  z_res <- z_functions(Theta_node, Theta_edge, N)
 
  grad_node <- S_node - n * z_res$grad_logZ_node
  grad_edge <- S_edge - n * z_res$grad_logZ_edge
 
  list(grad_node = grad_node, grad_edge = grad_edge)
}

 
soft_threshold <- function(x, tau) {
  sign(x) * pmax(abs(x) - tau, 0)
}

proximal_grad_ascent <- function(Theta_node_init, Theta_edge_init,
                                 Y_obs, N = 1000,
                                 step_size = 0.01, epsilon = 1e-4,
                                 max_iter = 500, lambda = 0,
                                 verbose = TRUE) {
  Theta_node <- Theta_node_init
  Theta_edge <- Theta_edge_init
  p <- nrow(Theta_node)
  M <- ncol(Theta_node)
 
  conv <- Inf
  iter <- 0
 
  while (conv > epsilon && iter < max_iter) {
    rate <- 1.0 / (1.0 + 0.01 * iter)
 
    g <- loglike_grad(Theta_node, Theta_edge, Y_obs, N)
 
    Theta_node_new <- Theta_node + rate * step_size * g$grad_node
    Theta_edge_new <- Theta_edge + rate * step_size * g$grad_edge
 
    Theta_edge_new <- (Theta_edge_new + t(Theta_edge_new)) / 2
    diag(Theta_edge_new) <- 0

    if (lambda > 0) {
      Theta_edge_new <- soft_threshold(Theta_edge_new, rate * step_size * lambda)
      diag(Theta_edge_new) <- 0
    }
 
    # conv <- sqrt(sum((Theta_node_new - Theta_node)^2) +
    #              sum((Theta_edge_new - Theta_edge)^2))
    conv <- sqrt((sum((Theta_node_new - Theta_node)^2) +
                 sum((Theta_edge_new - Theta_edge)^2))/(sum(Theta_node^2) + sum(Theta_edge^2)))
 
    Theta_node <- Theta_node_new
    Theta_edge <- Theta_edge_new
    iter <- iter + 1
 
    if (verbose && iter %% 50 == 0) {
      cat(sprintf("iter = %d, convergence = %.6f\n", iter, conv))
    }
  }
 
  if (verbose) {
    cat(sprintf("Stopped at iter %d, convergence = %.6f\n", iter, conv))
  }
 
  list(Theta_node = Theta_node, Theta_edge = Theta_edge, iter = iter, conv = conv)
}

library(VGAM)

fit_vgam <- function(Y_obs, M, symmetrize = "avg") {
  n <- nrow(Y_obs)
  p <- ncol(Y_obs)
 
  Theta_node <- matrix(0, nrow = p, ncol = M)
  Theta_edge <- matrix(0, nrow = p, ncol = p)
 
  for (s in 1:p) {
    cat(sprintf("  Fitting node %d / %d...\n", s, p))
 
    y_s <- ordered(Y_obs[, s], levels = 0:M)
    # covs <- as.data.frame(M - Y_obs[, -s, drop = FALSE]) / M
    covs <- as.data.frame(M - Y_obs[, -s, drop = FALSE])
    colnames(covs) <- paste0("V", (1:p)[-s])
 
    # Fit adjacent-category logit with shared slopes
    fit_s <- vglm(y_s ~ ., data = covs,
                  family = acat(parallel = TRUE, reverse = TRUE))
 
    cc <- coef(fit_s)
 
    intercept_names <- grep("^\\(Intercept\\)", names(cc), value = TRUE)
    Theta_node[s, ] <- cc[intercept_names]
 
    slope_names <- setdiff(names(cc), intercept_names)
    slopes <- cc[slope_names]
 
    others <- (1:p)[-s]
    for (k in seq_along(others)) {
      Theta_edge[s, others[k]] <- slopes[k]
    }
  }
 
  # Symmetrize
  if (symmetrize == "avg") {
    Theta_edge <- (Theta_edge + t(Theta_edge)) / 2
  } else if (symmetrize == "and") {
    mask <- (Theta_edge != 0) & (t(Theta_edge) != 0)
    Theta_edge <- (Theta_edge + t(Theta_edge)) / 2 * mask
  }
  diag(Theta_edge) <- 0
 
  list(Theta_node = Theta_node, Theta_edge = Theta_edge*(M^2))
}

library(ordinalNet)
.symmetrize_theta_edge <- function(Theta_edge, symmetrize = "avg") {
  if (symmetrize == "avg") {
    Theta_edge <- (Theta_edge + t(Theta_edge)) / 2
  } else if (symmetrize == "and") {
    mask <- (Theta_edge != 0) & (t(Theta_edge) != 0)
    Theta_edge <- (Theta_edge + t(Theta_edge)) / 2 * mask
  } else if (symmetrize != "none") {
    stop("symmetrize must be one of 'avg', 'and', or 'none'.")
  }

  diag(Theta_edge) <- 0
  Theta_edge
}

.select_ordinalNet_cv_lambda <- function(cvfit) {
  idx <- as.integer(cvfit$bestLambdaIndex)
  idx <- idx[!is.na(idx)]

  if (length(idx) == 0) {
    stop("ordinalNetCV did not return any selected lambda indices.")
  }

  tab <- tabulate(idx, nbins = length(cvfit$lambdaVals))
  which(tab == max(tab))[1]
}

.select_ordinalNet_tune_lambda <- function(tunefit, tuneMethod) {
  if (tuneMethod %in% c("cvLoglik", "cvDevPct")) {
    which.max(rowMeans(tunefit[[sub("^cv", "", tuneMethod)]], na.rm = TRUE))
  } else if (tuneMethod %in% c("cvMisclass", "cvBrier")) {
    which.min(rowMeans(tunefit[[sub("^cv", "", tuneMethod)]], na.rm = TRUE))
  } else {
    stop("Unsupported tuneMethod for ordinalNetTune.")
  }
}

fit_ordinalNet <- function(Y_obs, M, symmetrize = "avg",
                           alpha = 1, standardize = FALSE,
                           criteria = c("aic", "bic"),
                           which_lambda = NULL, ...) {
  if (!requireNamespace("ordinalNet", quietly = TRUE)) {
    stop("Package 'ordinalNet' is required for fit_ordinalNet().")
  }

  criteria <- match.arg(criteria)
  p <- ncol(Y_obs)

  Theta_node <- matrix(0, nrow = p, ncol = M)
  Theta_edge <- matrix(0, nrow = p, ncol = p)
  fit_list <- vector("list", p)
  lambda_index <- integer(p)
  lambda <- numeric(p)

  for (s in 1:p) {
    cat(sprintf("  Fitting node %d / %d with ordinalNet...\n", s, p))

    y_s <- ordered(Y_obs[, s], levels = 0:M)
    # x_s <- as.matrix((M - Y_obs[, -s, drop = FALSE]) / M)
    x_s <- as.matrix(M - Y_obs[, -s, drop = FALSE])
    colnames(x_s) <- paste0("V", (1:p)[-s])

    fit_s <- ordinalNet::ordinalNet(
      x = x_s,
      y = y_s,
      family = "acat",
      reverse = TRUE,
      parallelTerms = TRUE,
      nonparallelTerms = FALSE,
      alpha = alpha,
      standardize = standardize,
      ...
    )

    idx <- which_lambda
    if (is.null(idx)) {
      idx <- which.min(fit_s[[criteria]])
    }

    cc <- stats::coef(fit_s, whichLambda = idx, criteria = criteria)

    intercept_names <- grep("^\\(Intercept\\):", names(cc), value = TRUE)
    Theta_node[s, ] <- unname(cc[intercept_names])

    slope_names <- colnames(x_s)
    Theta_edge[s, (1:p)[-s]] <- unname(cc[slope_names])
    fit_list[[s]] <- fit_s
    lambda_index[s] <- idx
    lambda[s] <- fit_s$lambdaVals[idx]
  }

  Theta_edge <- .symmetrize_theta_edge(Theta_edge, symmetrize = symmetrize)

  list(
    Theta_node = Theta_node,
    Theta_edge = Theta_edge*(M^2),
    fit = fit_list,
    lambda_index = lambda_index,
    lambda = lambda
  )
}

fit_ordinalNet_cv <- function(Y_obs, M, symmetrize = "avg",
                              alpha = 1, standardize = FALSE,
                              tuneMethod = c("cvLoglik", "cvMisclass", "cvBrier",
                                             "cvDevPct", "aic", "bic"),
                              which_lambda = NULL, ...) {
  if (!requireNamespace("ordinalNet", quietly = TRUE)) {
    stop("Package 'ordinalNet' is required for fit_ordinalNet_cv().")
  }

  tuneMethod <- match.arg(tuneMethod)
  p <- ncol(Y_obs)

  Theta_node <- matrix(0, nrow = p, ncol = M)
  Theta_edge <- matrix(0, nrow = p, ncol = p)
  cvfit_list <- vector("list", p)
  lambda_index <- integer(p)
  lambda <- numeric(p)

  for (s in 1:p) {
    cat(sprintf("  Cross-validating node %d / %d with ordinalNet...\n", s, p))

    y_s <- ordered(Y_obs[, s], levels = 0:M)
    # x_s <- as.matrix((M - Y_obs[, -s, drop = FALSE]) / M)
    x_s <- as.matrix(M - Y_obs[, -s, drop = FALSE])
    colnames(x_s) <- paste0("V", (1:p)[-s])

    cvfit_s <- ordinalNet::ordinalNetCV(
      x = x_s,
      y = y_s,
      family = "acat",
      reverse = TRUE,
      parallelTerms = TRUE,
      nonparallelTerms = FALSE,
      alpha = alpha,
      standardize = standardize,
      tuneMethod = tuneMethod,
      ...
    )

    idx <- which_lambda
    if (is.null(idx)) {
      idx <- .select_ordinalNet_cv_lambda(cvfit_s)
    }

    cc <- stats::coef(cvfit_s$fit, whichLambda = idx)

    intercept_names <- grep("^\\(Intercept\\):", names(cc), value = TRUE)
    Theta_node[s, ] <- unname(cc[intercept_names])

    slope_names <- colnames(x_s)
    Theta_edge[s, (1:p)[-s]] <- unname(cc[slope_names])

    cvfit_list[[s]] <- cvfit_s
    lambda_index[s] <- idx
    lambda[s] <- cvfit_s$lambdaVals[idx]
  }

  Theta_edge <- .symmetrize_theta_edge(Theta_edge, symmetrize = symmetrize)

  list(
    Theta_node = Theta_node,
    Theta_edge = Theta_edge*(M^2),
    cvfit = cvfit_list,
    lambda_index = lambda_index,
    lambda = lambda
  )
}

lambda_max_node <- function(s, Y_obs, M) {
  n  <- nrow(Y_obs)
  U  <- (M - Y_obs)                # n x p
  u_s <- U[, s]
  Ut  <- U[, -s, drop = FALSE]
  r   <- u_s - mean(u_s)                 # centered response, length n
  max(abs(crossprod(Ut, r))) / n
}

fit_ordinalNet_cv_shared_lambda <- function(Y_obs, M, symmetrize = "avg",
                                            alpha = 1, standardize = FALSE,
                                            tuneMethod = c("cvLoglik", "cvMisclass",
                                                           "cvBrier", "cvDevPct"),
                                            lambdaVals = NULL,
                                            nFolds = 5,
                                            which_lambda = NULL,
                                            printProgress = TRUE,
                                            lambdaMinRatio = 0.01,
                                            nLambda = 10,
                                            ...) {
  if (!requireNamespace("ordinalNet", quietly = TRUE)) {
    stop("Package 'ordinalNet' is required for fit_ordinalNet_cv_shared_lambda().")
  }

  tuneMethod <- match.arg(tuneMethod)
  p <- ncol(Y_obs)

  if (is.null(lambdaVals)) {
    lambda_max_shared <- max(vapply(seq_len(p), lambda_max_node,
                                numeric(1), Y_obs = Y_obs, M = M))
    lambdaVals <- exp(seq(log(lambda_max_shared),
                        log(lambda_max_shared * lambdaMinRatio),
                        length.out = nLambda))
  }
  

  Theta_node <- matrix(0, nrow = p, ncol = M)
  Theta_edge <- matrix(0, nrow = p, ncol = p)
  tunefit_list <- vector("list", p)
  score_path <- matrix(NA_real_, nrow = length(lambdaVals), ncol = p)
  colnames(score_path) <- paste0("node", 1:p)

  metric_name <- switch(
    tuneMethod,
    cvLoglik = "loglik",
    cvMisclass = "misclass",
    cvBrier = "brier",
    cvDevPct = "devPct"
  )

  for (s in 1:p) {
    if (printProgress) {
      cat(sprintf("  Tuning node %d / %d with shared lambda path...\n", s, p))
    }

    y_s <- ordered(Y_obs[, s], levels = 0:M)
    x_s <- as.matrix(M - Y_obs[, -s, drop = FALSE])
    colnames(x_s) <- paste0("V", (1:p)[-s])

    tunefit_s <- ordinalNet::ordinalNetTune(
      x = x_s,
      y = y_s,
      lambdaVals = lambdaVals,
      nFolds = nFolds,
      family = "acat",
      reverse = TRUE,
      parallelTerms = TRUE,
      nonparallelTerms = FALSE,
      alpha = alpha,
      standardize = standardize,
      printProgress = FALSE,
      ...
    )

    score_path[, s] <- rowMeans(tunefit_s[[metric_name]], na.rm = TRUE)
    tunefit_list[[s]] <- tunefit_s
  }

  agg_score <- rowMeans(score_path, na.rm = TRUE)

  idx <- which_lambda
  if (is.null(idx)) {
    if (tuneMethod %in% c("cvLoglik", "cvDevPct")) {
      idx <- which.max(agg_score)
    } else {
      idx <- which.min(agg_score)
    }
  }

  for (s in 1:p) {
    x_names <- paste0("V", (1:p)[-s])
    cc <- stats::coef(tunefit_list[[s]]$fit, whichLambda = idx)

    intercept_names <- grep("^\\(Intercept\\):", names(cc), value = TRUE)
    Theta_node[s, ] <- unname(cc[intercept_names])
    Theta_edge[s, (1:p)[-s]] <- unname(cc[x_names])
  }

  Theta_edge <- .symmetrize_theta_edge(Theta_edge, symmetrize = symmetrize)

  list(
    Theta_node = Theta_node,
    Theta_edge = Theta_edge * (M^2),
    tunefit = tunefit_list,
    lambda_index = rep(idx, p),
    lambda = rep(lambdaVals[idx], p),
    lambda_shared = lambdaVals[idx],
    lambdaVals = lambdaVals,
    score_path = score_path,
    aggregated_score = agg_score,
    metric = metric_name
  )
}

frobenius_loss_theta <- function(Theta_node_hat, Theta_edge_hat,
                                 Theta_node_true, Theta_edge_true) {
  stopifnot(all(dim(Theta_node_hat) == dim(Theta_node_true)))
  stopifnot(all(dim(Theta_edge_hat) == dim(Theta_edge_true)))

  node_diff <- Theta_node_hat - Theta_node_true

  idx <- upper.tri(Theta_edge_true)
  edge_diff <- Theta_edge_hat[idx] - Theta_edge_true[idx]


  
  node_loss <- sum(node_diff^2)/sum(Theta_node_true^2)
  edge_loss <- sum(edge_diff^2)/sum(Theta_edge_true[idx]^2)
  total_loss <- (sum(node_diff^2) + sum(edge_diff^2))/(sum(Theta_node_true^2) + sum(Theta_edge_true[idx]^2))

  list(
    node_loss = node_loss,
    edge_loss = edge_loss,
    total_loss = total_loss
  )
}