function params = em( data, eps, k, maxIter, init )
% Expectation-Maximiation for Poisson-LDS model
% David Pfau 2012

addpath ../../util
addpath ../ssid % for block_hankel
addpath /Users/davidpfau/Documents/MATLAB/tprod
if nargin < 5 || strcmp(class(init),'double')
    if nargin < 5
        t = 0;
    else
        t = init;
    end
    [params map] = rand_init( k, size(data,1), size(data,2), t ); % initial estimate for phase path and parameters
else
    map = init.map;
    params = init.params;
end
[~,~,prec] = log_lik( data, map, params ); % get Hessian of log likelihood with crude parameter estimates
dat_ll_ = Inf;
dat_ll  = data_log_lik( data, map, prec, params );
i = 0;
fprintf('Iter \t E Data LL \t M Data LL \t Latent LL\n')
while i < maxIter %&& ( i < 10 || dat_ll - dat_ll_ < eps )
    i = i+1;
    [map,cll,prec] = newtons_method(@(x) log_lik( data, x, params ), map, 1e-8 ); % Initialize with path from previous step
    dat_ll__ = data_log_lik( data, map, prec, params );
    params = m_step( data, map, prec, params );
    dat_ll_ = dat_ll;
    dat_ll  = data_log_lik( data, map, prec, params );
    fprintf('%4i \t %2.4d \t %2.4d \t %2.4d \n',i,-dat_ll__,-dat_ll,-cll);
end