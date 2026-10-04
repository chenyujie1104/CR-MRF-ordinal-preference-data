// count_graphical_individual_parallel.cpp
//
// CR-MRF with individual-specific parameters, "parallel" (proportional-odds)
// node model.  Both the node and the edge parameters are linear in the
// covariates, with the node covariate effects shared across ordinal
// categories j:
//
//   theta_{s,j}(x_i) = alpha_{s,j} + sum_{v>=1} beta_{s,v} * x_{i,v}
//   theta_{st}(x_i)  = sum_v Theta_edge.slice(v)(s, t) * x_tilde(i, v)   (unchanged)
//
// Here:
//   * alpha_{s,j}  -- category-specific node intercepts (the cutpoints). These
//                     DO vary with the level j and live in slice 0 of Theta_node.
//   * beta_{s,v}   -- node covariate slopes. These do NOT vary with j; they are
//                     shared across all M categories.
//
// Contrast with the unrestricted model, where the node slopes theta_{s,j,v}
// are free for every (j, v).  Making the slopes parallel makes the
// joint MLE directly comparable to the pseudo-likelihood (ordinalNet) fit in
// fit_ordinalNet_individual(), which uses parallelTerms = TRUE,
// nonparallelTerms = FALSE: ordinalNet returns category-specific intercepts
// (acat_intercepts, p x M) and category-constant covariate slopes
// (intercept_slopes, p x q).  Those map one-to-one onto (alpha_{s,j}, beta_{s,v}).


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include <random>
using namespace Rcpp;

// ---------- helpers (duplicated from count_graphical.cpp) ----------

static arma::vec node_stat(const arma::vec& theta_s) {
  int M = theta_s.n_elem;
  arma::vec out(M + 1, arma::fill::zeros);
  double acc = 0.0;
  for (int j = M - 1; j >= 0; --j) {
    acc += theta_s(j);
    out(j) = acc;
  }
  return out; // out(M) = 0
}

static arma::vec node_pmf_indep(const arma::vec& theta_s) {
  arma::vec lu = node_stat(theta_s);
  lu -= lu.max();
  arma::vec u = arma::exp(lu);
  return u / arma::accu(u);
}

// Thread-safe categorical sampler: each thread owns its own mt19937_64
// (thread_local), seeded once from the OS entropy source, so no R RNG is
// touched and concurrent calls from OpenMP threads are safe.
static inline int sample_cat(const arma::vec& pmf) {
  static thread_local std::mt19937_64 rng(std::random_device{}());
  static thread_local std::uniform_real_distribution<double> unif(0.0, 1.0);
  double u = unif(rng);
  double cum = 0.0;
  int K = pmf.n_elem;
  for (int k = 0; k < K; ++k) {
    cum += pmf(k);
    if (u < cum) return k;
  }
  return K - 1;
}

// ---------- parallel-constraint helpers ----------

// Enforce the parallel constraint on a PARAMETER cube: for every covariate slice
// v >= 1, collapse the M columns to their mean and broadcast back, so each slope
// slice becomes column-constant (a single beta_{s,v} shared across categories).
// Slice 0 (the category-specific intercepts) is left untouched.
static void collapse_slope_param(arma::cube& Theta_node) {
  arma::uword M  = Theta_node.n_cols;
  arma::uword q1 = Theta_node.n_slices;
  for (arma::uword v = 1; v < q1; ++v) {
    arma::vec rowmean = arma::mean(Theta_node.slice(v), 1); // mean over j, length p
    for (arma::uword j = 0; j < M; ++j) Theta_node.slice(v).col(j) = rowmean;
  }
}

// Enforce the parallel constraint on a GRADIENT cube: for every covariate slice
// v >= 1, replace the M columns by their SUM over j (broadcast back).  This is
// d ell / d beta_{s,v} = sum_j d ell / d Theta_node(s,j,v).  Adding this collapsed
// gradient to a column-constant parameter slice keeps it column-constant.
// Slice 0 (intercepts) is left as-is (its per-j gradient is correct).
static void collapse_slope_grad(arma::cube& grad_node) {
  arma::uword M  = grad_node.n_cols;
  arma::uword q1 = grad_node.n_slices;
  for (arma::uword v = 1; v < q1; ++v) {
    arma::vec rowsum = arma::sum(grad_node.slice(v), 1); // sum over j, length p
    for (arma::uword j = 0; j < M; ++j) grad_node.slice(v).col(j) = rowsum;
  }
}

// Exported convenience wrappers, useful for sanity checks from R.
// [[Rcpp::export]]
arma::cube collapse_slope_param_parallel_cpp(arma::cube Theta_node) {
  collapse_slope_param(Theta_node);
  return Theta_node;
}
// [[Rcpp::export]]
arma::cube collapse_slope_grad_parallel_cpp(arma::cube grad_node) {
  collapse_slope_grad(grad_node);
  return grad_node;
}

// ---------- per-individual parameter construction ----------

// Build Tn_all (p, M, n) and Te_all (p, p, n) cubes whose slice i is the
// effective parameter matrix for individual i.  Identical to the non-parallel
// version: with a column-constant slope cube it yields
//   Tn_all.slice(i)(s, j) = alpha_{s,j} + sum_{v>=1} beta_{s,v} x_{i,v}.
static void make_individual_params(const arma::cube& Theta_node,
                                   const arma::cube& Theta_edge,
                                   const arma::mat&  X_tilde,
                                   arma::cube& Tn_all,
                                   arma::cube& Te_all) {
  int n  = X_tilde.n_rows;
  int q1 = X_tilde.n_cols;
  int p  = Theta_node.n_rows;
  int M  = Theta_node.n_cols;

  Tn_all.set_size(p, M, n);
  Te_all.set_size(p, p, n);

  for (int i = 0; i < n; ++i) {
    arma::mat Tn_i(p, M, arma::fill::zeros);
    arma::mat Te_i(p, p, arma::fill::zeros);
    for (int v = 0; v < q1; ++v) {
      double xv = X_tilde(i, v);
      if (xv != 0.0) {
        Tn_i += xv * Theta_node.slice(v);
        Te_i += xv * Theta_edge.slice(v);
      }
    }
    Tn_all.slice(i) = Tn_i;
    Te_all.slice(i) = Te_i;
  }
}

// [[Rcpp::export]]
Rcpp::List make_individual_params_parallel_cpp(const arma::cube& Theta_node,
                                               const arma::cube& Theta_edge,
                                               const arma::mat&  X_tilde) {
  arma::cube Tn_all, Te_all;
  make_individual_params(Theta_node, Theta_edge, X_tilde, Tn_all, Te_all);
  return Rcpp::List::create(_["Tn_all"] = Tn_all, _["Te_all"] = Te_all);
}

// ---------- independence sampler (one draw per individual) ----------

static arma::imat sample_independent_individual(const arma::cube& Tn_all) {
  int p = Tn_all.n_rows;
  int n = Tn_all.n_slices;
  arma::imat X(n, p);
  for (int i = 0; i < n; ++i) {
    const arma::mat& Tn_i = Tn_all.slice(i);
    for (int s = 0; s < p; ++s) {
      arma::vec pmf = node_pmf_indep(Tn_i.row(s).t());
      X(i, s) = sample_cat(pmf);
    }
  }
  return X;
}

// ---------- Gibbs sampler: n parallel chains, individual-specific params ----------

// [[Rcpp::export]]
arma::imat gibbs_cr_mrf_individual_parallel_cpp(const arma::cube& Theta_node,
                                                const arma::cube& Theta_edge,
                                                const arma::mat&  X_tilde,
                                                int n_iter = 500) {
  int p = Theta_node.n_rows;
  int M = Theta_node.n_cols;

  arma::cube Tn_all, Te_all;
  make_individual_params(Theta_node, Theta_edge, X_tilde, Tn_all, Te_all);

  int n = X_tilde.n_rows;
  arma::imat X = sample_independent_individual(Tn_all);

  // Precompute node_stat rows for each (s, i): cube (p, M+1, n).
  arma::cube NS(p, M + 1, n);
  for (int i = 0; i < n; ++i) {
    const arma::mat& Tn_i = Tn_all.slice(i);
    for (int s = 0; s < p; ++s) {
      NS.slice(i).row(s) = node_stat(Tn_i.row(s).t()).t();
    }
  }

  arma::vec scale(M + 1);
  for (int k = 0; k <= M; ++k) scale(k) = double(M - k) / double(M);

  arma::vec lu(M + 1), pmf(M + 1);

  for (int iter = 0; iter < n_iter; ++iter) {
    for (int s = 0; s < p; ++s) {
      for (int i = 0; i < n; ++i) {
        double r = 0.0;
        for (int t = 0; t < p; ++t) {
          if (t == s) continue;
          double th = Te_all(s, t, i);
          if (th != 0.0) r += th * double(M - X(i, t)) / double(M);
        }
        for (int k = 0; k <= M; ++k) lu(k) = NS(s, k, i) + r * scale(k);
        double m = lu.max();
        double S = 0.0;
        for (int k = 0; k <= M; ++k) { pmf(k) = std::exp(lu(k) - m); S += pmf(k); }
        pmf /= S;
        X(i, s) = sample_cat(pmf);
      }
    }
  }
  return X;
}

// ---------- IS for Z and moments (single individual) ----------

// Pure-Armadillo core (no Rcpp objects, no R API): safe to call from OpenMP
// worker threads.  Results are written into the output references.
static void z_moments_individual(const arma::mat& Tn_i,
                                 const arma::mat& Te_i,
                                 int N,
                                 double& log_Z_ratio,
                                 arma::mat& E_node,
                                 arma::mat& E_edge) {
  int p = Tn_i.n_rows;
  int M = Tn_i.n_cols;

  arma::imat Y(N, p);
  for (int s = 0; s < p; ++s) {
    arma::vec pmf = node_pmf_indep(Tn_i.row(s).t());
    for (int b = 0; b < N; ++b) Y(b, s) = sample_cat(pmf);
  }

  arma::mat MY(N, p);
  for (int b = 0; b < N; ++b)
    for (int s = 0; s < p; ++s)
      MY(b, s) = double(M - Y(b, s)) / double(M);

  arma::mat TMY = MY * Te_i;
  arma::vec q_vals(N);
  for (int b = 0; b < N; ++b)
    q_vals(b) = 0.5 * arma::dot(TMY.row(b), MY.row(b));

  double q_max = q_vals.max();
  arma::vec d = arma::exp(q_vals - q_max);
  double sum_d = arma::accu(d);
  log_Z_ratio = q_max + std::log(sum_d / double(N));

  E_node.zeros(p, M);
  for (int j = 1; j <= M; ++j) {
    for (int s = 0; s < p; ++s) {
      double acc = 0.0;
      for (int b = 0; b < N; ++b) if (Y(b, s) <= j - 1) acc += d(b);
      E_node(s, j - 1) = acc / sum_d;
    }
  }

  arma::mat dMY = MY;
  for (int b = 0; b < N; ++b) dMY.row(b) *= d(b);
  E_edge = (dMY.t() * MY) / sum_d;
}

// [[Rcpp::export]]
Rcpp::List z_moments_individual_parallel_cpp(const arma::mat& Tn_i,
                                             const arma::mat& Te_i,
                                             int N) {
  double log_Z_ratio;
  arma::mat E_node, E_edge;
  z_moments_individual(Tn_i, Te_i, N, log_Z_ratio, E_node, E_edge);
  return Rcpp::List::create(
    _["log_Z_ratio"] = log_Z_ratio,
    _["E_node"]      = E_node,
    _["E_edge"]      = E_edge
  );
}

// ---------- log-likelihood gradient (parallel node slopes) ----------

// [[Rcpp::export]]
Rcpp::List loglike_grad_individual_parallel_cpp(const arma::cube& Theta_node,
                                                const arma::cube& Theta_edge,
                                                const arma::mat&  X_tilde,
                                                const arma::imat& Y_obs,
                                                int N = 1000) {
  int n  = Y_obs.n_rows;
  int p  = Y_obs.n_cols;
  int M  = Theta_node.n_cols;
  int q1 = X_tilde.n_cols;

  arma::cube Tn_all, Te_all;
  make_individual_params(Theta_node, Theta_edge, X_tilde, Tn_all, Te_all);

  arma::cube grad_node(p, M, q1, arma::fill::zeros);
  arma::cube grad_edge(p, p, q1, arma::fill::zeros);

  // Parallel over individuals: each thread accumulates into its own local
  // cubes (no Rcpp / R API calls inside), merged once at the end.
#pragma omp parallel
  {
    arma::cube gn_loc(p, M, q1, arma::fill::zeros);
    arma::cube ge_loc(p, p, q1, arma::fill::zeros);

#pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
      arma::mat Tn_i = Tn_all.slice(i);
      arma::mat Te_i = Te_all.slice(i);

      arma::mat S_node_i(p, M, arma::fill::zeros);
      for (int j = 1; j <= M; ++j)
        for (int s = 0; s < p; ++s)
          if (Y_obs(i, s) <= j - 1) S_node_i(s, j - 1) = 1.0;

      arma::rowvec my_i(p);
      for (int s = 0; s < p; ++s) my_i(s) = double(M - Y_obs(i, s)) / double(M);
      arma::mat S_edge_i = my_i.t() * my_i;

      double log_Z_ratio;
      arma::mat E_node, E_edge;
      z_moments_individual(Tn_i, Te_i, N, log_Z_ratio, E_node, E_edge);

      arma::mat D_node_i = S_node_i - E_node;
      arma::mat D_edge_i = S_edge_i - E_edge;

      for (int v = 0; v < q1; ++v) {
        double xv = X_tilde(i, v);
        if (xv != 0.0) {
          gn_loc.slice(v) += xv * D_node_i;
          ge_loc.slice(v) += xv * D_edge_i;
        }
      }
    }

#pragma omp critical
    {
      grad_node += gn_loc;
      grad_edge += ge_loc;
    }
  }
  double inv_n = 1.0 / double(n);
  grad_node *= inv_n;
  grad_edge *= inv_n;

  // Parallel constraint: collapse the covariate-slope slices to their column-sum
  // over j.  Slice 0 (category-specific intercepts) is left per-j.
  collapse_slope_grad(grad_node);

  return Rcpp::List::create(
    _["grad_node"] = grad_node,
    _["grad_edge"] = grad_edge
  );
}

// ---------- approximate log-likelihood ----------

static double log_Z0_individual(const arma::mat& Tn_i) {
  int p = Tn_i.n_rows;
  double out = 0.0;
  for (int s = 0; s < p; ++s) {
    arma::vec ns = node_stat(Tn_i.row(s).t());
    double m = ns.max();
    out += m + std::log(arma::accu(arma::exp(ns - m)));
  }
  return out;
}

// Works unchanged on the parallel cube (column-constant slope slices yield the
// correct theta_{s,j}(x_i) = alpha_{s,j} + sum_v beta_{s,v} x_{i,v}).
// [[Rcpp::export]]
double approx_loglik_individual_parallel_cpp(const arma::cube& Theta_node,
                                             const arma::cube& Theta_edge,
                                             const arma::mat&  X_tilde,
                                             const arma::imat& Y_obs,
                                             int N_track = 5000) {
  int n = Y_obs.n_rows;
  int p = Y_obs.n_cols;
  int M = Theta_node.n_cols;

  arma::cube Tn_all, Te_all;
  make_individual_params(Theta_node, Theta_edge, X_tilde, Tn_all, Te_all);

  double ll = 0.0;
  // Parallel over individuals; per-i contributions are independent, so a
  // simple sum reduction suffices (no Rcpp / R API calls inside).
#pragma omp parallel for schedule(static) reduction(+:ll)
  for (int i = 0; i < n; ++i) {
    arma::mat Tn_i = Tn_all.slice(i);
    arma::mat Te_i = Te_all.slice(i);

    double lin_node = 0.0;
    for (int j = 1; j <= M; ++j)
      for (int s = 0; s < p; ++s)
        if (Y_obs(i, s) <= j - 1) lin_node += Tn_i(s, j - 1);

    arma::rowvec my_i(p);
    for (int s = 0; s < p; ++s) my_i(s) = double(M - Y_obs(i, s)) / double(M);
    double lin_edge = 0.5 * arma::as_scalar(my_i * Te_i * my_i.t());

    double log_Z0_i = log_Z0_individual(Tn_i);
    double log_ratio;
    arma::mat E_node_i, E_edge_i;
    z_moments_individual(Tn_i, Te_i, N_track, log_ratio, E_node_i, E_edge_i);
    double logZ_i = log_Z0_i + log_ratio;

    ll += lin_node + lin_edge - logZ_i;
  }
  return ll;
}

// ---------- proximal operators (edge group lasso; unchanged) ----------

static void soft_thresh_group_cube(arma::cube& C, double tau, double w) {
  arma::uword p  = C.n_rows;
  arma::uword pc = C.n_cols;
  arma::uword q1 = C.n_slices;
  double tau_w   = tau * w;

  arma::mat fibre_norms(p, pc, arma::fill::zeros);
  for (arma::uword v = 0; v < q1; ++v) fibre_norms += arma::square(C.slice(v));
  fibre_norms = arma::sqrt(fibre_norms);

  arma::mat shrink(p, pc);
  for (arma::uword s = 0; s < p; ++s)
    for (arma::uword t = 0; t < pc; ++t) {
      double fn = fibre_norms(s, t);
      shrink(s, t) = (fn > tau_w) ? (1.0 - tau_w / fn) : 0.0;
    }
  for (arma::uword v = 0; v < q1; ++v) C.slice(v) %= shrink;
}

// ---------- proximal gradient ascent (parallel node slopes) ----------

// [[Rcpp::export]]
Rcpp::List proximal_grad_ascent_individual_parallel_cpp(
    const arma::cube&  Theta_node_init,
    const arma::cube&  Theta_edge_init,
    const arma::mat&   X_tilde,
    const arma::imat&  Y_obs,
    int          N           = 1000,
    double       step_size   = 0.01,
    double       epsilon     = 1e-4,
    int          max_iter    = 500,
    double       lambda      = 0.0,
    bool         track_obj   = false,
    int          track_every = 50,
    int          N_track     = 5000,
    bool         verbose     = true) {
  arma::cube Theta_node = Theta_node_init;
  arma::cube Theta_edge = Theta_edge_init;
  int q1 = Theta_node.n_slices;

  // Make sure the initial slope slices satisfy the parallel constraint.
  collapse_slope_param(Theta_node);

  std::vector<int>    obj_iters;
  std::vector<double> obj_values;
  if (track_obj) {
    obj_iters.push_back(0);
    obj_values.push_back(approx_loglik_individual_parallel_cpp(
      Theta_node, Theta_edge, X_tilde, Y_obs, N_track));
  }

  double conv = std::numeric_limits<double>::infinity();
  int iter = 0;

  while (conv > epsilon && iter < max_iter) {
    double rate = 1.0 / (1.0 + 0.01 * double(iter));

    // Gradient is already collapsed onto the parallel subspace inside
    // loglike_grad_individual_parallel_cpp, so the slope slices of Tn_new stay
    // column-constant after the ascent step.
    Rcpp::List g = loglike_grad_individual_parallel_cpp(
      Theta_node, Theta_edge, X_tilde, Y_obs, N);
    arma::cube gn = Rcpp::as<arma::cube>(g["grad_node"]);
    arma::cube ge = Rcpp::as<arma::cube>(g["grad_edge"]);

    arma::cube Tn_new = Theta_node + rate * step_size * gn;
    arma::cube Te_new = Theta_edge + rate * step_size * ge;

    // Symmetrize every edge slice; zero diagonals.
    for (int v = 0; v < q1; ++v) {
      arma::mat S = Te_new.slice(v);
      S = 0.5 * (S + S.t());
      S.diag().zeros();
      Te_new.slice(v) = S;
    }

    // Proximal step on edge parameters (group lasso).
    if (lambda > 0.0) {
      double tau = rate * step_size * lambda;
      double w = std::sqrt(double(q1));
      soft_thresh_group_cube(Te_new, tau, w);
      for (int v = 0; v < q1; ++v) Te_new.slice(v).diag().zeros();
    }

    // Guard against numerical drift: re-impose the parallel constraint exactly.
    collapse_slope_param(Tn_new);

    double d1  = arma::accu(arma::square(Tn_new - Theta_node));
    double d2  = arma::accu(arma::square(Te_new - Theta_edge));
    double nrm = arma::accu(arma::square(Theta_node))
               + arma::accu(arma::square(Theta_edge));
    conv = std::sqrt((d1 + d2) / std::max(nrm, 1e-12));

    Theta_node = Tn_new;
    Theta_edge = Te_new;
    ++iter;

    if (track_obj && (iter % track_every == 0)) {
      double ll = approx_loglik_individual_parallel_cpp(
        Theta_node, Theta_edge, X_tilde, Y_obs, N_track);
      obj_iters.push_back(iter);
      obj_values.push_back(ll);
      if (verbose)
        Rcpp::Rcout << "iter = " << iter
                    << ", conv = " << conv
                    << ", approx loglik = " << ll << "\n";
    } else if (verbose && (iter % 50 == 0)) {
      Rcpp::Rcout << "iter = " << iter << ", convergence = " << conv << "\n";
    }
  }

  if (verbose)
    Rcpp::Rcout << "Stopped at iter " << iter << ", convergence = " << conv << "\n";

  // Compact parallel parameterization, kept ALONGSIDE the full cube above.
  //   alpha : p x M  category-specific intercepts (cutpoints) = slice 0.
  //   beta  : p x q  shared node slopes; column v-1 is the (column-constant)
  //           slope slice v, so we read its first column.  Empty when q == 0.
  arma::mat alpha = Theta_node.slice(0);
  arma::mat beta(Theta_node.n_rows, q1 - 1);
  for (int v = 1; v < q1; ++v) beta.col(v - 1) = Theta_node.slice(v).col(0);

  Rcpp::List out = Rcpp::List::create(
    _["Theta_node"] = Theta_node,
    _["Theta_edge"] = Theta_edge,
    _["alpha"]      = alpha,
    _["beta"]       = beta,
    _["iter"]       = iter,
    _["conv"]       = conv,
    _["step_size"]  = step_size,
    _["N"]          = N,
    _["lambda"]     = lambda,
    _["epsilon"]    = epsilon
  );
  if (track_obj) {
    out["obj_iters"]  = Rcpp::wrap(obj_iters);
    out["obj_values"] = Rcpp::wrap(obj_values);
  }
  return out;
}
