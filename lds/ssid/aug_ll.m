function [y grad Hinfo] = aug_ll(x,x1,lam,rho,dat,z)
% Objective function/gradient/fields necessary for multiplication by the
% Hessian for minimizing the augmented log likelihood term in ADMM

N = size(x,2) - 1;
yh = x(:,1:N);
b = x(:,end);
eyb = exp( yh + b*ones(1,N) );

y = lam*sum( sum( eyb - dat.*( yh + b*ones(1,N) ) ) ) ...
    + 0.5*rho*( (x1(:) - x(:) + z(:))'*(x1(:) - x(:) + z(:)) );
grad = lam*( [eyb - dat, sum( eyb - dat, 2 )] ) + rho*( x - x1 - z );
Hinfo = struct('lam',lam,'rho',rho,'eyb',eyb);