function x = conj_grad( linop, b, eps, mode, precond, x0 )
% Solves the linear system b = linop( x ) for x where linop is a symmetric
% linear operator by conjugate gradient descent, with preconditioning

if nargin < 3
    eps = 1e-12;
end
if nargin < 4
    mode = 0;
end
if nargin < 5
    precond = @(x) x;
end
if nargin < 6
    x = zeros(size(b));
else
    x = x0;
end

r = b - linop(x);
p = r;
Mr = precond( r );
r_ = r(:)'*Mr(:);
i = 0;
while sqrt(r_) > eps
    i = i + 1;
    if mode == 1, fprintf('Iter %d, sqrt(r_): %d, Residual: %d\n', i, sqrt(r_), norm(b - linop(x))); end
    Ap = linop(p);
    a = r_/(p(:)'*Ap(:));
    x = x + a*p;
    r = r - a*Ap;
    Mr = precond( r );
    r__ = r(:)'*Mr(:);
    p = Mr + (r__/r_)*p;
    r_ = r__;
end