function [moesp_params n4sid_params s] = ssid( y, u, i, opts )
% Merge N4SID and MOESP into a single file for recovering parameters of a
% state space model both ways following whatever preprocessing steps like
% nuclear norm minimization we might be interested in.  opts struct is the
% same as in moesp.m, and documentation can be found there.

l = size( y, 1 );
m = size( u, 1 );
N = size( y, 2 );
assert( size( u, 2 ) == N, 'Input and output do not have the same number of time steps' );
opts = default_opts( opts );

Oi = build_proj( y, u, i, opts );
[r,s,v] = svd( Oi );
n = find( diag( s )/s(1) < opts.tol, 1 ) - 1; % approximate order of the system
if isempty( n )
    n = opts.maxOrder;
else
    n = min( n, opts.maxOrder );
end

%% N4SID

X  = sqrt( s( 1:n, 1:n ) ) * v( :, 1:n )';

ABCD = [X(:,2:end); y(:,1:size(X,2)-1)] * pinv( [X(:,1:end-1); u(:,1:size(X,2)-1)] );
A = stabilize( ABCD( 1:n, 1:n ) );
B = ABCD( 1:n, n+1:end );
C = ABCD( n+1:end, 1:n );
D = ABCD( n+1:end, n+1:end );

resid = [X(:,2:end); Yii] - ABCD*[X(:,1:end-1); Uii];
covar = resid * resid';
Q = covar( 1:n, 1:n );
R = covar( n+1:end, n+1:end );
S = covar( 1:n, n+1:end );

n4sid_params = struct('A',A,'B',B,'C',C,'D',D,'Q',Q,'R',R,'S',S,'X',X);

%% MOESP

G = r( :, 1:n ) * sqrt( s( 1:n, 1:n ) );
C = G( 1:l, : );
A = pinv( G( 1:(i-1)*l, : ) ) * G( (l+1):i*l, : );
A = stabilize( A );

F1 = zeros( l*N, n );
F1( 1:l, : ) = C;
for t = 1:N-1
    F1( t*l + (1:l), : ) = F1( (t-1)*l + (1:l), : ) * A;
end
if opts.instant, F2 = kron( u(:,1:N)', eye(l) ); end
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

if opts.instant
    xx = pinv( [F1, F2, F3], 1e-6 ) * y( 1:l*N )';
    x0 = xx(1:n);
    D = reshape( xx( n + (1:l*m) ), l, m );
    B = reshape( xx(n + l*m + 1:end ), n, m );
else
    xx = pinv( [F1 F3], 1e-6 ) * y( 1:l*N )';
    x0 = xx(1:n);
    B = reshape( xx(n+1:end), n, m );
    D = zeros(l,m);
end

moesp_params = struct('A',A,'B',B,'C',C,'D',D,'x0',x0);

s = diag(s);