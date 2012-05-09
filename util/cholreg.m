function L = cholreg( A, e, n )
% Regularized Cholesky decomposition.  Trying it doesn't seem to make much
% difference on computing the log determinant, which is all we're using it
% for.

if nargin < 3
    n = 0;
    if nargin < 2
        e = 1e-3;
    end
end

if n < 100
    try
        L = chol(A + n*e*eye(size(A,1)));
    catch
        L = cholreg(A,e,n+1);
    end
else
    [L,~] = chol(A);
end