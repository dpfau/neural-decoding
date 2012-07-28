function x = min_res( linop, b, eps, mode, x0 )
% Solves the linear system b = linop( x ) for x where linop is a symmetric
% linear operator by the minimum residual algorithm

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
Ar = linop( r );
r_ = Ar(:)'*r(:);
i = 0;
while sqrt(r_) > eps
    i = i + 1;
    if mode == 1, fprintf('Iter %d, sqrt(r_): %d, Residual: %d\n', i, sqrt(r_), norm( b - linop( x ) ) ); end
    Ap = linop( p );
    a = r_/(Ap(:)'*Ap(:));
    x = x + a*p;
    r = r - a*Ap;
    Ar = linop( r );
    r__ = Ar(:)'*r(:);
    p = r + (r__/r_)*p;
    r_ = r__;
end