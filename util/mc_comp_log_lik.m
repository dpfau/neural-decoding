function ll = mc_comp_log_lik( data, map, prec, params, t )
% Generate Monte Carlo samples from Laplace approximation of latent state
% posterior and calculate their likelihood

z = map(:)*ones(1,t) + chol(sparse_hess(prec))\randn(numel(map),t);
ll = zeros(t,1);
for i = 1:t
    ll(i) = log_lik( data, reshape(z(:,i),size(map)), params );
end