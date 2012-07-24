function x = conj_grad_underdet( linop, b )
% Solves the underdetermined linear system b = linop( x ) for x where
% linop is a linear operator in TFOCS format.  Used for initializing TFOCS 
% at a feasible point.  Specifically, it solves the system
% A*A'*u = b, where A*x = linop( x ) and A' is the adjoint of A, by linear
% conjugate gradient, and then sets x = A'*u

dim = linop( [], 0 );
assert( dim{2}(1) == size( b, 1 ) && dim{2}(2) == size( b, 2 ), 'Output size does not match operator' );
AAt = @(x) linop( linop( x, 2 ), 1 ); 
u = conj_grad( AAt, b, 1e-12, 1 );
x = linop( u, 2 );