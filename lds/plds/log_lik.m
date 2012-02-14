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
Cxb = add_vector( params.C*map, params.b );
fCxb  = params.f( Cxb, 0 );
fCxb1 = params.f( Cxb, 1 );
Qinv = params.Q^-1;

ll = 1/2*sum( sum( mapdiff.*(Qinv*mapdiff) ) ) ...
     + sum( sum( fCxb ) ) ...
     - sum( sum( data.*log( fCxb ) ) );
 
grad = [zeros(size(mapdiff,1),1), Qinv*mapdiff(:,2:end)] ...
     - [params.A'*Qinv*mapdiff(:,1:end-1), zeros(size(mapdiff,1),1)] ...
     + params.C'*fCxb1 + params.C'*(data.*fCxb1./fCxb);
Hinfo = struct('diag',,'off_diag',);