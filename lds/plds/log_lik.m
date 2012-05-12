function [ll grad Hinfo] = log_lik( data, map, params )
% Log likelihood and relevent gradient/Hessian wrt latent variable for
% Poisson-LDS model:
% data(t) ~ Poiss( f( C*map(t) + b ) )
% map(t+1) ~ N(A*map(t),Q)
% Note that in fact everything is wrt the *negative* log likelihood,
% because in general we want to maximize these quantities, but the built in
% Matlab functions are for minimizing.  Also, Hinfo is then the precision,
% not negative precision, which makes things slightly easier.
%
% Input:
%   data - observed data
%   map - inferred path through the latent states
%   params - parameter estimates for PLDS system, containing fields:
%       A - linear latent state evolution matrix
%       Q - latent state noise covariance
%       C - linear latent-state-to-intensity mapping
%       b - bias before nonlinear mapping
%       f - nonlinear mapping to Poisson intensity, or pointwise derivative
%           if passed a second argument (f(x,k) is the kth derivative of f
%           at x)
%       x0 - initial latent state mean
%       Q0 - initial latent state covariance
%
% Output:
%   ll - log likelihood
%   grad - gradient of ll wrt map
%   Hinfo - struct containing two field, diag and off_diag, that are the
%       diagonal and off-diagonal of the Hessian of ll wrt map, which is
%       tridiagional
%
% David Pfau, 2012

mapdiff = map(:,2:end) - params.A*map(:,1:end-1);
CxbDy = [params.C, params.b, params.D]*[map; ones(1,size(map,2)); zeros(size(params.D,2),params.t), block_hankel(data,1,params.t,size(data,2)-1)];
f0 = params.f( CxbDy, 0 );
f1 = params.f( CxbDy, 1 );
f2 = params.f( CxbDy, 2 );

datalogf0 = data.*log( f0 );
datalogf0(isnan(datalogf0)) = 0; % adopt the 0*log(0) = 0 convention
ll = 1/2*trace(params.Q\(mapdiff*mapdiff')) + (size(map,2)-1)/2*log(det(params.Q)) ...
     + 1/2*(map(:,1)-params.x0)'*(params.Q0\(map(:,1)-params.x0)) + 1/2*log(det(params.Q0)) ...
     + numel(map)/2*log(2*pi) ...
     + sum( sum( f0 ) ) ...
     - sum( sum( datalogf0 ) ) ...
     + sum( sum( gammaln( data + 1 ) ) );
 
grad = [params.Q0\(map(:,1)-params.x0), params.Q\mapdiff] ...
      - [params.A'*(params.Q\mapdiff), zeros(size(mapdiff,1),1)] ...
      + params.C'*f1 - params.C'*(data.*f1./f0);
 
Hinfo = struct('diag_corner', params.Q0^-1, ...
               'diag_upper',  params.A'*(params.Q\params.A), ... 
               'diag_lower',  params.Q^-1, ...
               'diag_left',   params.C', ...
               'diag_center', f2 - data.*(f2.*f0 - f1.^2)./f0.^2, ...
               'diag_right',  params.C, ...
               'off_diag',   -params.A'/params.Q);
grad(isnan(grad)) = 0;
Hinfo.diag_center(isnan(Hinfo.diag_center)) = 0;