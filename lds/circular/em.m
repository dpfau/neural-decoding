function params = em( data, eps, thresh, up, N, maxIter )
% Expectation-Maximiation for latent-phase model
% David Pfau 2012

addpath ../../util
if nargin < 3
    figure(1); plot( data );
    R = input('Enter threshold\n','s');
    thresh = str2double(R);
    R = input('Enter direction of threshold crossing\n','s');
    up = str2double(R);
    R = input('Enter template size\n','s');
    N = str2double(R);
end

map = make_init( data, thresh, up, .1 ); % initial estimate for phase path
params = m_step( data, N, map ); % very crude parameter estimates
[~,~,prec] = log_lik( data, map, params ); % get Hessian of log likelihood with crude parameter estimates

fe = Inf;
fe_ = exp_comp_log_lik( data, map, prec, params ) - entropy( prec );
i = 0;
while i < maxIter && ( i < 10 || fe - fe_ > eps )
    i = i+1;
    fe = fe_;
    [map,prec,pll] = e_step( data, params, map ); % initialize with path from previous step
    [params,ecll,fe_] = m_step( data, N, map, prec );
    fprintf('Iter: %4i\tFE: %d\tECLL: %d\tPhase LL: %d\n',i,fe_,ecll,pll);
end