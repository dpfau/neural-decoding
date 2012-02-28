function [A B C D X Q R S s] = n4sid( y, u, i, opts )

opts = default_opts( opts );
Oi = build_proj( y, u, i, opts );
[~,s,v] = svd( Oi );
n = find( diag( s )/s(1) < opts.tol, 1 ) - 1; % approximate order of the system
if isempty( n )
    n = opts.maxOrder;
else
    n = min( n, opts.maxOrder );
end
X  = sqrt( s( 1:n, 1:n ) ) * v( :, 1:n )';

ABCD = [X(:,2:end); y(:,1:size(X,2)-1)] * pinv( [X(:,1:end-1); u(:,1:size(X,2)-1)] );
A = stabilize( ABCD( 1:n, 1:n ) );
B = ABCD( 1:n, n+1:end );
C = ABCD( n+1:end, 1:n );
D = ABCD( n+1:end, n+1:end );

resid = [X(:,2:end); y(:,1:size(X,2)-1)] - ABCD*[X(:,1:end-1); u(:,1:size(X,2)-1)];
covar = resid * resid';
Q = covar( 1:n, 1:n );
R = covar( n+1:end, n+1:end );
S = covar( 1:n, n+1:end );
s = diag(s);