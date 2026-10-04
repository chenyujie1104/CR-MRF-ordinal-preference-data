// cr_mrf.cpp
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// ---------- helpers ----------

// node_stat(theta_s)[k] = sum_{j=k+1}^{M} theta_{s;j},  k = 0,...,M   (length M+1)
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

// Draw one category in {0,...,M} by inverse CDF from a PMF of length M+1.
static inline int sample_cat(const arma::vec& pmf) {
  double u = R::unif_rand();
  double cum = 0.0;
  int K = pmf.n_elem;
  for (int k = 0; k < K; ++k) {
    cum += pmf(k);
    if (u < cum) return k;
  }
  return K - 1;
}

// ---------- independence sampler ----------

// [[Rcpp::export]]
arma::imat sample_independent_cpp(const arma::mat& Theta_node, int n) {
  int p = Theta_node.n_rows;
  arma::imat X(n, p);
  for (int s = 0; s < p; ++s) {
    arma::vec pmf = node_pmf_indep(Theta_node.row(s).t());
    for (int i = 0; i < n; ++i) X(i, s) = sample_cat(pmf);
  }
  return X;
}

// ---------- Gibbs: n parallel chains, return final states ----------

// [[Rcpp::export]]
arma::imat gibbs_cr_mrf_cpp(const arma::mat& Theta_node,
                            const arma::mat& Theta_edge,
                            int n, int n_iter = 500) {
  int p = Theta_node.n_rows;
  int M = Theta_node.n_cols;

  arma::imat X = sample_independent_cpp(Theta_node, n);

  // Precompute node_stat rows (p x (M+1))
  arma::mat NS(p, M + 1);
  for (int s = 0; s < p; ++s)
    NS.row(s) = node_stat(Theta_node.row(s).t()).t();

  // Precompute (M - k)/M for k = 0..M
  arma::vec scale(M + 1);
  for (int k = 0; k <= M; ++k) scale(k) = double(M - k) / double(M);

  arma::vec lu(M + 1), pmf(M + 1);

  for (int iter = 0; iter < n_iter; ++iter) {
    for (int s = 0; s < p; ++s) {
      for (int i = 0; i < n; ++i) {
        // r_s = sum_{t != s} theta_{st} (M - X_{i,t})/M
        double r = 0.0;
        for (int t = 0; t < p; ++t) {
          double th = Theta_edge(s, t);
          if (th != 0.0 && t != s)
            r += th * double(M - X(i, t)) / double(M);
        }
        // log-unnormalized conditional
        for (int k = 0; k <= M; ++k) lu(k) = NS(s, k) + r * scale(k);
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

// ---------- importance sampling for Z and grad log Z ----------

// [[Rcpp::export]]
Rcpp::List z_functions_cpp(const arma::mat& Theta_node,
                           const arma::mat& Theta_edge,
                           int N) {
  int p = Theta_node.n_rows;
  int M = Theta_node.n_cols;

  arma::imat Y = sample_independent_cpp(Theta_node, N);

  // MY_{b,s} = (M - Y_{b,s}) / M  in [0,1]
  arma::mat MY(N, p);
  for (int b = 0; b < N; ++b)
    for (int s = 0; s < p; ++s)
      MY(b, s) = double(M - Y(b, s)) / double(M);

  // q_b = (1/2) MY_b^T Theta_edge MY_b
  arma::mat TMY = MY * Theta_edge;                // N x p
  arma::vec q_vals(N);
  for (int b = 0; b < N; ++b)
    q_vals(b) = 0.5 * arma::dot(TMY.row(b), MY.row(b));

  double q_max = q_vals.max();
  arma::vec d = arma::exp(q_vals - q_max);        // centered weights
  double sum_d = arma::accu(d);
  double Z_ratio = std::exp(q_max) * sum_d / double(N);

  // Node gradient: grad_node(s, j-1) = sum_b d_b * 1{Y_{b,s} <= j-1} / sum_d
  arma::mat grad_node(p, M, arma::fill::zeros);
  for (int j = 1; j <= M; ++j) {
    for (int s = 0; s < p; ++s) {
      double acc = 0.0;
      for (int b = 0; b < N; ++b)
        if (Y(b, s) <= j - 1) acc += d(b);
      grad_node(s, j - 1) = acc / sum_d;
    }
  }

  // Edge gradient: (diag(d) MY)^T MY / sum_d
  arma::mat dMY = MY;
  for (int b = 0; b < N; ++b) dMY.row(b) *= d(b);
  arma::mat grad_edge = (dMY.t() * MY) / sum_d;

  return Rcpp::List::create(
    _["Z_ratio"]        = Z_ratio,
    _["grad_logZ_node"] = grad_node,
    _["grad_logZ_edge"] = grad_edge
  );
}

// ---------- log-likelihood gradient ----------

// [[Rcpp::export]]
Rcpp::List loglike_grad_cpp(const arma::mat& Theta_node,
                            const arma::mat& Theta_edge,
                            const arma::imat& Y_obs,
                            int N) {
  int n = Y_obs.n_rows;
  int p = Y_obs.n_cols;
  int M = Theta_node.n_cols;

  // Observed node sufficient statistic: S_node(s, j-1) = #{i : Y_{i,s} <= j-1}
  arma::mat S_node(p, M, arma::fill::zeros);
  for (int j = 1; j <= M; ++j) {
    for (int s = 0; s < p; ++s) {
      int c = 0;
      for (int i = 0; i < n; ++i) if (Y_obs(i, s) <= j - 1) ++c;
      S_node(s, j - 1) = double(c);
    }
  }

  // Observed edge sufficient statistic (normalized): sum_i (M-Y_s)(M-Y_t)/M^2
  arma::mat MY_obs(n, p);
  for (int i = 0; i < n; ++i)
    for (int s = 0; s < p; ++s)
      MY_obs(i, s) = double(M - Y_obs(i, s)) / double(M);
  arma::mat S_edge = MY_obs.t() * MY_obs;

  Rcpp::List z = z_functions_cpp(Theta_node, Theta_edge, N);
  arma::mat gZn = Rcpp::as<arma::mat>(z["grad_logZ_node"]);
  arma::mat gZe = Rcpp::as<arma::mat>(z["grad_logZ_edge"]);

  double inv_n = 1.0 / double(n);

  return Rcpp::List::create(
    _["grad_node"] = inv_n * S_node - gZn,
    _["grad_edge"] = inv_n * S_edge - gZe
  );
}

// ---------- proximal gradient ascent with l1 penalty on edges ----------

static inline double soft_thresh_scalar(double x, double tau) {
  double a = std::abs(x) - tau;
  if (a <= 0.0) return 0.0;
  return (x > 0.0 ? 1.0 : -1.0) * a;
}

static double log_Z0(const arma::mat& Theta_node) {
  int p = Theta_node.n_rows;
  double out = 0.0;
  for (int s = 0; s < p; ++s) {
    arma::vec ns = node_stat(Theta_node.row(s).t());  // length M+1
    double m = ns.max();
    out += m + std::log(arma::accu(arma::exp(ns - m)));
  }
  return out;
}

static double approx_loglik(const arma::mat& Theta_node,
                            const arma::mat& Theta_edge,
                            const arma::mat& S_node,
                            const arma::mat& S_edge,
                            int n, int N_track) {
  int p = Theta_node.n_rows;
  int M = Theta_node.n_cols;

  arma::imat Y = sample_independent_cpp(Theta_node, N_track);
  arma::mat MY(N_track, p);
  for (int b = 0; b < N_track; ++b)
    for (int s = 0; s < p; ++s)
      MY(b, s) = double(M - Y(b, s)) / double(M);

  arma::mat TMY = MY * Theta_edge;
  arma::vec q(N_track);
  for (int b = 0; b < N_track; ++b)
    q(b) = 0.5 * arma::dot(TMY.row(b), MY.row(b));

  double q_max = q.max();
  double log_ratio = q_max + std::log(arma::accu(arma::exp(q - q_max)) / double(N_track));
  double logZ = log_Z0(Theta_node) + log_ratio;

  double lin = arma::dot(Theta_node, S_node) + 0.5 * arma::dot(Theta_edge, S_edge);
  return lin - double(n) * logZ;
}

// [[Rcpp::export]]
Rcpp::List proximal_grad_ascent_cpp(const arma::mat& Theta_node_init,
                                    const arma::mat& Theta_edge_init,
                                    const arma::imat& Y_obs,
                                    int N = 1000,
                                    double step_size = 0.01,
                                    double epsilon = 1e-4,
                                    int max_iter = 500,
                                    double lambda = 0.0,
                                    bool track_obj = false,
                                    int track_every = 50,
                                    int N_track = 5000,
                                    bool verbose = true) {
  arma::mat Theta_node = Theta_node_init;
  arma::mat Theta_edge = Theta_edge_init;
  int n = Y_obs.n_rows;
  int p = Y_obs.n_cols;
  int M = Theta_node.n_cols;

  // Precompute observed sufficient statistics (only needed if tracking)
  arma::mat S_node, S_edge;
  if (track_obj) {
    S_node.zeros(p, M);
    for (int j = 1; j <= M; ++j)
      for (int s = 0; s < p; ++s) {
        int c = 0;
        for (int i = 0; i < n; ++i) if (Y_obs(i, s) <= j - 1) ++c;
        S_node(s, j - 1) = double(c);
      }
    arma::mat MY_obs(n, p);
    for (int i = 0; i < n; ++i)
      for (int s = 0; s < p; ++s)
        MY_obs(i, s) = double(M - Y_obs(i, s)) / double(M);
    S_edge = MY_obs.t() * MY_obs;
  }

  std::vector<int>    obj_iters;
  std::vector<double> obj_values;

  if (track_obj) {
    obj_iters.push_back(0);
    obj_values.push_back(approx_loglik(Theta_node, Theta_edge, S_node, S_edge, n, N_track));
  }

  double conv = std::numeric_limits<double>::infinity();
  int iter = 0;

  while (conv > epsilon && iter < max_iter) {
    double rate = 1.0 / (1.0 + 0.01 * double(iter));

    Rcpp::List g = loglike_grad_cpp(Theta_node, Theta_edge, Y_obs, N);
    arma::mat gn = Rcpp::as<arma::mat>(g["grad_node"]);
    arma::mat ge = Rcpp::as<arma::mat>(g["grad_edge"]);

    arma::mat Tn_new = Theta_node + rate * step_size * gn;
    arma::mat Te_new = Theta_edge + rate * step_size * ge;

    Te_new = 0.5 * (Te_new + Te_new.t());
    Te_new.diag().zeros();

    if (lambda > 0.0) {
      double tau = rate * step_size * lambda;
      for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j)
          Te_new(i, j) = soft_thresh_scalar(Te_new(i, j), tau);
      Te_new.diag().zeros();
    }

    double d1 = arma::accu(arma::square(Tn_new - Theta_node));
    double d2 = arma::accu(arma::square(Te_new - Theta_edge));
    conv = std::sqrt((d1 + d2)/(arma::accu(arma::square(Theta_node)) + arma::accu(arma::square(Theta_edge))));

    Theta_node = Tn_new;
    Theta_edge = Te_new;
    ++iter;

    // Periodic approximate log-likelihood tracking
    if (track_obj && (iter % track_every == 0)) {
      double ll = approx_loglik(Theta_node, Theta_edge, S_node, S_edge, n, N_track);
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

  Rcpp::List out = Rcpp::List::create(
    _["Theta_node"] = Theta_node,
    _["Theta_edge"] = Theta_edge,
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

// [[Rcpp::export]]
double approx_loglik_cpp(const arma::mat& Theta_node,
                         const arma::mat& Theta_edge,
                         const arma::imat& Y_obs,
                         int N_track = 5000) {
  int n = Y_obs.n_rows;
  int p = Y_obs.n_cols;
  int M = Theta_node.n_cols;

  // Observed sufficient statistics
  arma::mat S_node(p, M, arma::fill::zeros);
  for (int j = 1; j <= M; ++j)
    for (int s = 0; s < p; ++s) {
      int c = 0;
      for (int i = 0; i < n; ++i) if (Y_obs(i, s) <= j - 1) ++c;
      S_node(s, j - 1) = double(c);
    }

  arma::mat MY_obs(n, p);
  for (int i = 0; i < n; ++i)
    for (int s = 0; s < p; ++s)
      MY_obs(i, s) = double(M - Y_obs(i, s)) / double(M);
  arma::mat S_edge = MY_obs.t() * MY_obs;

  // IS estimate of log(Z / Z_0)
  arma::imat Y = sample_independent_cpp(Theta_node, N_track);
  arma::mat MY(N_track, p);
  for (int b = 0; b < N_track; ++b)
    for (int s = 0; s < p; ++s)
      MY(b, s) = double(M - Y(b, s)) / double(M);

  arma::mat TMY = MY * Theta_edge;
  arma::vec q(N_track);
  for (int b = 0; b < N_track; ++b)
    q(b) = 0.5 * arma::dot(TMY.row(b), MY.row(b));

  double q_max = q.max();
  double log_ratio = q_max + std::log(arma::accu(arma::exp(q - q_max)) / double(N_track));
  double logZ = log_Z0(Theta_node) + log_ratio;

  double lin = arma::dot(Theta_node, S_node) + 0.5 * arma::dot(Theta_edge, S_edge);
  return lin - double(n) * logZ;
}
