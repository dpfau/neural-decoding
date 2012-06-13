function p1 = conj_grad( f, grad, init, eps, mode )
% Conjugate gradient minimization of a function
% f    - function handle of function to be minimized
% grad - function handle of gradient of function to be minimized
% init - initial guess
% eps  - threshold change in objective function at which minimization stops
% David Pfau, 2011

p0 = init;
g0 = grad( init );
h0 = g0;
dx = Inf;

if mode, fprintf( 'Iter \t f(x) \t\t || grad(x) || \n' ), end
i = 0;
while abs( dx ) > eps
    if mode, fprintf( '%i \t %4.2d \t %4.2 \n', i, f( p0 ), abs( dx ) ), end
    i = i+1;
    dx = fminbnd( @(x) f( p0 + x * h0 ), -1e6, 1e6 );
    p1 = p0 + dx * h0;
    g1 = grad( p1 );
    if ( g1 == zeros( size( g1 ) ) )
        break
    end
    h1 = g1 + ( g1(:)' * ( g1(:) - g0(:) ) ) / ( g1(:)' * g1(:) ) * h0;
    
    p0 = p1;
    g0 = g1;
    h0 = h1;
end