function [A B C D x0 s B1 D1 x1] = moesp( y, u, i, N, opts )
% Subspace identification for the linear time invariant system:
% x(t+1) = A*x(t) + B*u(t)
% y(t) = C*x(t) + D*u(t)
% For data with output noise, one approach following Zhang and Vandenberghe
% 2010 is to use nuclear norm minimization.  To use this approach with this
% code, TFOCS is required: http://tfocs.stanford.edu/
%
% Mandatory inputs:
% y - output data, one column per time step
% u - input data, one column per time step
% i - number of block-Hankel rows.  i*l should be greater than system order
% N - number of timesteps from the data used in reconstruction
%
% Fields in opts:
% noise - type of output noise
%   - none - default
%   - gauss - Gaussian noise, use nuclear norm minimization
%   - poiss - Poisson noise, use nuclear norm minimization + ADMM
% proj - 
%   - orth_svd - orthogonal projection computed from SVD
%   - orth_pinv - orthogonal projection computed as I - pinv(U)*U
%   - oblique - oblique projection, as used in standard MOESP
% tol - the ratio between the greatest singular value and the last one
%   used for choosing the system order
% maxOrder - the maximum possible system order
% instant - if 1, the matrix D is included in the recovered model, if 0, D
%   is set to 0
% vsig - the tradeoff between output log likelihood and nuclear norm, if
%   opts.noise ~= 'none'
% rho - the constant factor for the augmented lagrangian in ADMM, if
%   opts.noise == 'poiss'
% tfocs_path - path to TFOCS
%
% David Pfau, 2011-2012

l = size( y, 1 );
m = size( u, 1 );
opts = default_opts( opts );

%% Reconstruct A, C
[Oi yh] = build_proj( y, u, i, opts );
[r,s,~] = svd( Oi );
n = find( diag( s )/s(1) < opts.tol, 1 ) - 1; % approximate order of the system
if isempty( n )
    n = opts.maxOrder;
else
    n = min( n, opts.maxOrder );
end

G = r( :, 1:n ) * sqrt( s( 1:n, 1:n ) );
C = G( 1:l, : );
A = stabilize( pinv( G( 1:(i-1)*l, : ) ) * G( (l+1):i*l, : ) );

%% Crazy least-squares problem to recover x0, B and D simultaneously
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
    
    xx = pinv( [F1, F2, F3], 1e-6 ) * yh( 1:l*N )';
    x1 = xx(1:n);
    D1 = reshape( xx( n + (1:l*m) ), l, m );
    B1 = reshape( xx(n + l*m + 1:end ), n, m );
else
    xx = pinv( [F1 F3], 1e-6 ) * y( 1:l*N )';
    x0 = xx(1:n);
    B = reshape( xx(n+1:end), n, m );
    D = zeros(l,m);
    
    xx = pinv( [F1 F3], 1e-6 ) * yh( 1:l*N )';
    x1 = xx(1:n);
    B1 = reshape( xx(n+1:end), n, m );
    D1 = zeros(l,m);
end
s = diag(s);