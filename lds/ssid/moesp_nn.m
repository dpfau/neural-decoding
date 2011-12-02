function [A B C D x0 s s0] = moesp_nn( y, u, i, N, vsig, eps )
addpath /Users/davidpfau/Documents/MATLAB/TFOCS
if nargin < 6
    eps = 1e-3;
end

l = size( y, 1 );
m = size( u, 1 );

U = block_hankel( u, 1, i, N );
assert( size(U,1) < size(U,2) );
[~,~,v] = svd(U);
Un = v(:,m*i+1:end);
s0 = svd( block_hankel( y, 1, i, N ) * Un );

%% Nuclear norm minimization using TFOCS, then take svd of YU^\perp
opts = tfocs_SCD;
opts.tol = 1e-4; % don't have all day here, folks...
opts.printEvery = 10;
yh = tfocs_SCD( [], ...
    @(varargin) hankel_op( Un, l, i, N, varargin{:} ), ...
    @proj_spectral, ...
    2*s0(1)/l/N/vsig^2, ...
    y(:,1:N), ...
    hankel_op( Un, l, i, N, y(:,1:N), 1 ), ...
    opts );
[r,s,~] = svd( hankel_op( Un, l, i, N, yh, 1 ) );

%% Approximate order of the system
n = find( diag( s )/s(1) < eps, 1 ) - 1; 
if isempty( n ), n = 10; end

%% Recover A, C
G = r( :, 1:n ) * sqrt( s( 1:n, 1:n ) );
C = G( 1:l, : );
A = pinv( G( 1:(i-1)*l, : ) ) * G( (l+1):i*l, : );
A = stabilize( A );

%% Crazy least-squares problem to recover x0, B and D simultaneously
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