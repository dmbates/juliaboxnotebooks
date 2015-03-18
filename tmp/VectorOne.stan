data {
  int<lower=0>  N; // num observations
  int<lower=1>  K; // length of fixed-effects vector
  int<lower=1>  M; // num subjects
  int<lower=1>  J; // length of individual vector-valued random effects
  int<lower=1,upper=M> subj[N]; // subject indicator
  matrix[N,K]   X; // model matrix for fixed-effects parameters
  row_vector[J] Z[N]; // generator model matrix for random-effects
  vector[N]     y; // response vector (reaction time)
}

parameters {
  cholesky_factor_corr[J] L; // Cholesky factor of unconditional correlation of random effects
  vector[J] tau;  // relative standard deviations of unconditional distribution of random effects
  vector[J] u[M]; // spherical random effects
  vector[K] beta; // fixed-effects
  real<lower=0> sigma; // standard deviation of response given random effects
}

transformed parameters {
  matrix[J,J] Lambda; 
  matrix[J,J] corr;
  vector[J] b[M];
  vector[N] muX;
  vector[N] mu;
  corr <- tcrossprod(L);  // for monitoring the correlations
  Lambda <- diag_pre_multiply(tau,L);
  for (m in 1:M)
    b[m] <-  Lambda * u[m];
  muX <- X * beta;
  for (n in 1:N)
    mu[n] <- muX[n] + Z[n] * b[subj[n]];
}

model {
  tau ~ cauchy(0,2.5);
  L ~ lkj_corr_cholesky(2);
  for (m in 1:M)
    u[m] ~ normal(0,sigma);
  y ~ normal(mu, sigma);
}