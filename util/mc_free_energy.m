function fe = mc_free_energy( data, map, prec, params, t )
% Monte Carlo integral to approximate EM free energy

z = map(:)*ones(1,t) + chol(sparse_hess(prec))\randn(numel(map),t);
fe = zeros(t,1);
for i = 1:t
    fe(i) = log_lik( data, reshape(z(:,i),size(map)), params ) ...
        - 1/2*( (z(:,i)-map(:))'*sparse_hess(prec)*(z(:,i)-map(:)) + numel(map)*log(2*pi) - log_det_tridiag( prec ) );
end