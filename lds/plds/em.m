function params = em( data, eps, k, maxIter )
% Expectation-Maximiation for Poisson-LDS model
% David Pfau 2012

addpath ../../util
addpath /Users/davidpfau/Documents/MATLAB/tprod
[map params] = make_init( data, k ); % initial estimate for phase path and parameters
[~,~,prec] = log_lik( data, map, params ); % get Hessian of log likelihood with crude parameter estimates

fe = Inf;
fe_ = exp_comp_log_lik( data, map, prec, params ) - entropy( prec );
i = 0;
while i < maxIter && ( i < 10 || fe - fe_ > eps )
    i = i+1;
    fe = fe_;
    [map,prec,xll] = e_step( data, params, map ); % initialize with path from previous step
    [params,ecll,fe_] = m_step( data, map, prec, params.f );
    fprintf('Iter: %4i\tFE: %d\tECLL: %d\tLatent LL: %d\n',i,fe_,ecll,xll);
end