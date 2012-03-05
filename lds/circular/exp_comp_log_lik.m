function ecll = exp_comp_log_lik( data, map, prec, params )
% The expected complete log likelihood, that is, the quantity maximized in
% the M-step of EM.  The expectation is taken with respect to the Laplace
% approximation of the posterior probability of the latent path, with mean
% map and (tridiagonal) precision matrix prec
%
% Input -
%   data: the observed data
%   map: the MAP latent phase path, computed in E step
%   prec: struct with two fields, diag and off_diag, which give the
%       diagonal and off-diagonal of the tridiagonal precision matrix for 
%       the latent phase path, also computed during E step.
%   params: struct with the parameters of the latent phase model
%
% Output -
%   ecll: expected complete log likelihood
%
% David Pfau, 2012

N = length(params.template);
T = length(data);
[extt extt1] = prod_expectation( map, prec );
[ey ey2] = exp_interp( map, prec, params.template ); % Expected data and data squared
ecll = ( -extt(1)/2 ...
       - extt(end)/2 ...
       - sum(extt(2:end-1)) ...
       + map(end)*params.dt ...
       - map(1)*params.dt ...
       + sum(extt1) ...
       - (T-1)*params.dt^2/2 )/params.sig_dt^2 ...
       + ( -sum(ey2)/2 + ey*data' - data*data'/2 )/params.sig_temp^2 ...
       - T*log(params.sig_dt) - T*log(params.sig_temp) - T*log(2*pi);