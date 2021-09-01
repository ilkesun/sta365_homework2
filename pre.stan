data {
  int<lower=0> N;
  real lower_bound;
  real upper_bound;
  int<lower = 0> J_major; 
  int<lower = 0 > J_gender; 
  vector<lower = lower_bound, upper = upper_bound>[N]anxiety_before;
  int<lower = 1, upper = J_major> major[N];
  int<lower = 1, upper = J_gender> gender[N];
  int<lower=0, upper=1> only_prior;
}

parameters {
  real mu;
  real<lower=0> tau_major;
  vector<multiplier = tau_major>[J_major] u_major;
  real<lower=0> tau_gender;
  vector<multiplier = tau_gender>[J_gender] u_gender; 
  real<lower=0> sigma;
}

transformed parameters {
  vector [N]eq = mu + u_gender[gender] + u_major[major];
}

model {
  mu ~ normal(0, 5); 
  sigma ~ normal(0, 5);
  u_major ~ normal(0, tau_major);
  tau_major ~ normal(0, 5);
  u_gender ~ normal(0, tau_gender);
  tau_gender ~ normal(0, 5);
  
  target += normal_lpdf(anxiety_before | eq, sigma) - log_diff_exp(normal_lcdf(
    upper_bound | eq, sigma), normal_lcdf(lower_bound | eq, sigma));
            
  if(only_prior == 0) {
    anxiety_before ~ normal(eq, sigma);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] anxiety_pred;
  for (i in 1:N) {
    log_lik[i] = normal_lpdf(anxiety_before[i]| eq[i], sigma);
    anxiety_pred[i] = normal_rng(eq[i], sigma);
  }
}

