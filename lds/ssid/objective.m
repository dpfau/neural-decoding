function obj = objective(yh,yh1,b1,y,lam,i,Un)
% Augmented lagrangian objective for subspace ID with non-Gaussian output
% noise

l = size(yh,1);
N = size(yh,2);
obj = sum( svd( hankel_op( Un, l, i, N, yh, 1 ) ) ) ...
    + lam*sum( sum( exp( yh1 + b1*ones(1,N) ) - y.*( yh1 + b1*ones(1,N) ) ) );