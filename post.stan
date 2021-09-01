data {
  int<lower=0> N;
  real lower_bound;
  real upper_bound;
  vector<lower = lower_bound, upper = upper_bound>[N]anxiety_before;
  vector<lower=0, upper=1>[N] Z;
  int<lower = 0> J_major; 
  int<lower = 0 > J_gender; 
  vector<lower = lower_bound, upper = upper_bound>[N]anxiety_after;
  int<lower = 1, upper = J_major> major[N];
  int<lower = 1, upper = J_gender> gender[N];
  int<lower=0, upper=1> only_prior;
}

parameters {
  real mu;
  real beta1;
  real beta2;
  real<lower=0> tau_major;
  vector<multiplier = tau_major>[J_major] u_major;
  real<lower=0> tau_gender;
  vector<multiplier = tau_gender>[J_gender] u_gender;
  real<lower=0> sigma;
}

transformed parameters {
  vector [N]eq = mu + u_gender[gender] + u_major[major] + beta1*anxiety_before 
  + beta2*Z;
}

model {
  mu ~ normal(0, 5); 
  sigma ~ normal(0, 5);
  u_major ~ normal(0, tau_major);
  tau_major ~ normal(0, 3);
  u_gender ~ normal(0, tau_gender);
  tau_gender ~ normal(0, 3);
  
  target += normal_lpdf(anxiety_after | eq, sigma) - 
  log_diff_exp(normal_lcdf(upper_bound | eq, sigma), 
  normal_lcdf(lower_bound | eq, sigma));
  
  if(only_prior == 0) {
    anxiety_after ~ normal(eq, sigma);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] anxiety_pred;
  for (i in 1:N) {
    log_lik[i] = normal_lpdf(anxiety_after[i]| eq[i], sigma);
    anxiety_pred[i] = normal_rng(eq[i], sigma);
  }
}
