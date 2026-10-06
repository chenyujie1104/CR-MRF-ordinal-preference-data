# Covariate-dependent Joint Modeling of Multivariate Ordinal Preferences and Its Connections with Comparison Models - Supplementary Code

This repository contains the core implementation and example scripts for fitting covariate-dependent consecutive-ratio Markov random fields (CR-MRF) to multivariate ordinal preference data.

## Contents

- **`src/count_graphical.cpp`**: C++ implementation of the base CR-MRF, with parameters shared across individuals
- **`src/count_graphical_individual_parallel.cpp`**: C++ implementation of the covariate CR-MRF, with individual-specific parameters with free, category-specific covariate slopes.
- **`R/count_graphical_cr_mrf.R`**: R implementation of the base model and shared helpers (`node_stat`, `node_pmf_indep`, `soft_threshold`, samplers)
- **`R/count_graphical_individual_model.R`**: R helpers for the covariate model, including the node-wise pseudo-likelihood (`ordinalNet`, `VGAM`) comparison fits
- **`Application/MovieLens/example_run.R`**: Example R script demonstrating how to fit the model and reproduce the MovieLens results
- **`Application/MovieLens/Movie1M/selected5_reviewer_matrix.csv`**: MovieLens 1M data block, 234 reviewers x 5 movies with reviewer covariates
- **`Application/MovieLens/Movie1M/selected5_movies.csv`**: Titles, years, genres and rating counts for the five movies

## Prerequisites

### Required R Packages
```r
install.packages(c("Rcpp", "RcppArmadillo", "ggplot2", "doParallel", "ordinalNet", "foreach", "VGAM"))
```

### System Requirements
- R (version 4.0.0 or higher recommended; tested with 4.2.2)
- C++ compiler with C++11 support, with OpenMP support for parallel computing
- RcppArmadillo and its dependencies

### Parallel Computing Setup

Two parts of the code run in parallel. Both are optional: the example runs
single-threaded as well, only slower.

**OpenMP (the joint MLE).** The gradient of the log normalizing constant is
accumulated in parallel over individuals, and the log-likelihood evaluation is
parallelised the same way. `// [[Rcpp::plugins(openmp)]]` in the C++ source
makes `sourceCpp()` compile with OpenMP enabled, so there is nothing to
configure at compile time.

**Setting the number of threads:**

Export `OMP_NUM_THREADS` in the shell before starting R:

```bash
export OMP_NUM_THREADS=4      # adjust based on your system's cores
Rscript example_run.R
```


**foreach / doParallel (the pseudo-likelihood fit).**
`fit_ordinalNet_individual()` fits the `p` node-wise regressions through
`%dopar%`, so it needs a registered backend. `example_run.R` registers one
worker:

```r
library(doParallel)
registerDoParallel(1)    # up to p (= 5 here) to fit the nodes concurrently
```

With no backend registered, `foreach` runs sequentially and warns about it.

**Note:** Without parallel computing enabled the runtime is significantly
longer. 

## Quick Start

1. **Navigate to the example directory:**
```r
setwd("path/to/cr-mrf/Application/MovieLens")
```

2. **Run the example script:**
```bash
Rscript example_run.R
```

The paths inside the script are relative to that directory, so it has to be run from there; both outputs are written to the same place.

The example script will:
- Load the MovieLens 1M data block (234 reviewers, 5 movies, ratings recoded to levels 0-4)
- Fit the covariate CR-MRF by joint MLE on the 170 training reviewers
- Simulate the 64 held-out reviewers from the fitted model by Gibbs sampling
- Fit the node-wise MPLE (`ordinalNet`) and a covariate-dependent Plackett-Luce model on the same training data
- Reproduce the node-slope table (Table S.4) and the pairwise-preference comparison (Figure 1 and Figure S.3) in the paper up to Monte Carlo variation
- Save outputs to your local directory

The run takes a couple of minutes with several OpenMP threads available and considerably longer single-threaded (see Parallel Computing Setup).

## Main Functions

### 1. **Joint MLE by proximal gradient ascent** (`proximal_grad_ascent_individual_parallel_cpp`)
- Required: `Theta_node_init`, `Theta_edge_init`, `X_tilde`, `Y_obs`
- Parameters: `N`, `step_size`, `epsilon`, `max_iter`, `lambda`, `verbose`

### 2. **Gibbs sampler for the fitted model** (`gibbs_cr_mrf_individual_parallel_cpp`)
- Required: `Theta_node`, `Theta_edge`, `X_tilde`
- Parameters: `n_iter`

### 3. **Node-wise pseudo-likelihood fit** (`fit_ordinalNet_individual`)
- Required: `Y_obs`, `X`, `M`
- Parameters: `symmetrize`, `alpha`, `standardize`, `criteria`, `which_lambda`,
  plus arguments forwarded to `ordinalNet` (the example passes `lambdaVals = 0`)

## Usage

### Basic Usage

To call the model on your own data (paths below are relative to the
repository root):

```r
library(Rcpp)
library(RcppArmadillo)

# Compile the C++ code
Rcpp::sourceCpp("src/count_graphical_individual_parallel.cpp")

# Import helper functions (pseudo-likelihood comparison fits)
source("R/count_graphical_individual_model.R")

# Your data: Y_obs is an n x p integer matrix of ordinal levels 0..M;
# X_tilde is n x (q + 1), the covariates with a leading column of ones
X_tilde <- cbind(1, x1, x2)

# Set up parameters
p <- ncol(Y_obs)        # number of nodes
M <- 4                  # highest ordinal level (levels are 0..M)
q1 <- ncol(X_tilde)     # covariates, intercept included
Theta_node_init <- array(-0.1, dim = c(p, M, q1))
Theta_edge_init <- array(-0.1 * (1 - diag(p)), dim = c(p, p, q1))

# Fit the joint MLE
fit <- proximal_grad_ascent_individual_parallel_cpp(
  Theta_node_init, Theta_edge_init, X_tilde, Y_obs,
  N = 10000, step_size = 0.1, max_iter = 3000, lambda = 0.0, verbose = TRUE)

# Simulate new individuals with covariates X_new_tilde
draws <- gibbs_cr_mrf_individual_parallel_cpp(
  fit$Theta_node, fit$Theta_edge, X_new_tilde, n_iter = 2000)
```

### Parameter Tuning Guide

- **`step_size`**: proximal gradient step size
  - Start with 0.1 and reduce if the convergence trace is not monotone
- **`N`**: importance-sampling draws per individual per iteration, used for the
  gradient of the log normalizing constant
  - Higher values: less gradient noise but slower. Default in the example: 10000
- **`epsilon`**: convergence tolerance on the relative change in the
  parameters between iterations
  - Default: 1e-4. The example stops at ~1300 iterations
- **`max_iter`**: iteration cap if `epsilon` is not reached
- **`lambda`**: group-lasso penalty on the edge parameters, each edge's
  covariate slices forming one group, so an edge is dropped for all covariates
  at once
  - Controls sparsity of the graph. `lambda = 0` gives the unpenalized joint MLE
- **`n_iter`** (sampler): Gibbs sweeps per chain
  - The example repeats each covariate row K = 500 times and runs 2000 sweeps

## Output Structure

`proximal_grad_ascent_individual_parallel_cpp` returns a list containing:

- **`Theta_node`**: p x M x (q+1) cube of node parameters. Slice 1 holds the
  category-specific intercepts; slices 2.. hold the covariate slopes
- **`Theta_edge`**: p x p x (q+1) cube of edge parameters, each slice symmetric
  with a zero diagonal
- **`alpha`**: p x M matrix of category-specific intercepts (slice 1 of `Theta_node`)
- **`beta`**: p x q matrix of node slopes, one column per covariate. Slopes are
  shared across rating levels in this parallel model, so this is the form the
  node-slope table reports
- **Convergence information**: `iter` and `conv`, the relative parameter change at the last iteration
- **Settings used**: `step_size`, `N`, `lambda`, `epsilon`

`gibbs_cr_mrf_individual_parallel_cpp` returns an n x p integer matrix of
simulated ordinal levels, one row per row of `X_tilde`.

## Data

The five movies were picked by a greedy shared-audience search, so every
reviewer in the block rated all five and the response matrix is dense, with no
missing cells. Ratings are 1-5 stars in the csv and recoded to levels 0-4 by
the script, so `M = 4`. Reviewer covariates are `age`, `age_code`, `age_group`,
`gender`, `occupation` and `zip_code`. `Age` is the midpoint of the reviewer's
MovieLens age group, which lets it serve as a continuous covariate, and the
example uses standardised age and a female indicator.

The raw MovieLens 1M corpus is not redistributed here. It is available from
<https://grouplens.org/datasets/movielens/1m/>.


