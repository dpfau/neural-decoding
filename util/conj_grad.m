function x = conj_grad( linop, b, eps, mode, x0 )
% Solves the linear system b = linop( x ) for x where linop is a symmetric
% linear operator

if nargin == 3
    mode = 0;
end
if nargin < 5
    x = zeros( size( b ) );
else
    x = x0;
end
r = b - linop( x );
p = r;
r_ = r(:)'*r(:);
i = 0;
while sqrt(r_) > eps
    i = i + 1;
    if mode == 1, fprintf('Iter %d, Residual: %d\n', i, sqrt(r_)); end
    Ap = linop( p );
    a = r_/(p(:)'*Ap(:));
    x = x + a*p;
    r = r - a*Ap;
    r__ = r(:)'*r(:);
    p = r + (r__/r_)*p;
    r_ = r__;
end