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
Cxb = [params.C, params.b]*[map; ones(1,size(map,2))];
fCxb  = params.f( Cxb, 0 );
fCxb1 = params.f( Cxb, 1 );
fCxb2 = params.f( Cxb, 2 );
Qinv = params.Q^-1;

ll = 1/2*sum( sum( mapdiff.*(Qinv*mapdiff) ) ) ...
     + sum( sum( fCxb ) ) ...
     - sum( sum( data.*log( fCxb ) ) );

grad = [zeros(size(mapdiff,1),1), Qinv*mapdiff] ...
     - [params.A'*Qinv*mapdiff, zeros(size(mapdiff,1),1)] ...
     + params.C'*fCxb1 - params.C'*(data.*fCxb1./fCxb);
 
Hinfo = struct('diag_upper',  params.A'*Qinv*params.A, ...
               'diag_lower',  Qinv, ...
               'diag_left',   params.C', ...
               'diag_center', fCxb2 - data.*(fCxb2.*fCxb - fCxb1.^2)./fCxb.^2, ...
               'diag_right',  params.C, ...
               'off_diag',   -params.A'*Qinv);