function [A B C D x0 s] = moesp4( y, u, i, N, eps )
% Reconstruct B, D and x0 simultaneously, and use oblique rather than
% orthogonal projection

if nargin < 5
    eps = 1e-3;
end

l = size( y, 1 );
m = size( u, 1 );

Y = block_hankel( y, 1, 2*i, N );
U = block_hankel( u, 1, 2*i, N );

Yf = Y(i+1:end,:);
Uf = U(i+1:end,:);

% Oi = Y * ( eye( N ) - pinv( U ) * U );
Oi = oblique( Yf, Uf, [Y(1:i,:);U(1:i,:)] );

[r,s,~] = svd( Oi );
n = find( diag( s )/s(1) < eps, 1 ) - 1; % approximate order of the system
if isempty( n ), n = 10; end
G = r( :, 1:n ) * sqrt( s( 1:n, 1:n ) );
C = G( 1:l, : );
A = pinv( G( 1:(i-1)*l, : ) ) * G( (l+1):i*l, : );

% Stabilize A
[UA,TA] = schur( A, 'complex' );
eigs = diag(TA);
ns = nnz( abs( eigs ) > 1 );
while ns > 0
    eigs( abs( eigs ) > 1 ) = 1./eigs( abs( eigs ) > 1 );
    TA( 1:n+1:end ) = eigs;
    A = real( UA*TA*UA' );
    [UA,TA] = schur( A, 'complex' );
    eigs = diag(TA);
    ns = nnz( abs( eigs ) > 1 );
end

% Crazy least-squares problem to recover x0, B and D simultaneously
F1 = zeros( l*N, n );
F1( 1:l, : ) = C;
for t = 1:N-1
    F1( t*l + (1:l), : ) = F1( (t-1)*l + (1:l), : ) * A;
end
F2 = kron( u(:,1:N)', eye(l) );
F3 = zeros( l*N, n*m );
for ii = 1:N-1
    F3t = zeros( l*(N-ii), n*m );
    for jj = 1:l
        for kk = 1:m
            F3t( jj + (0:l:l*(N-ii-1)), (kk-1)*n + (1:n) ) = u( kk, 1:N-ii )' * F1( (ii-1)*l + jj, : );
        end
    end
    F3( ii*l + 1:end, : ) = F3( ii*l + 1:end, : ) + F3t;
end

xx = pinv( [F1, F2, F3], 1e-6 ) * y( 1:l*N )';
x0 = xx(1:n);
D = reshape( xx( n + (1:l*m) ), l, m );
B = reshape( xx(n + l*m + 1:end ), n, m );
s = diag(s);