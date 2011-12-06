function [A B C D x0 b s s0] = moesp_nn_poiss_out( y, u, i, N, vsig, rho, eps )
% Learns a Wiener model with known output nonlinearity and Poisson noise by
% nuclear norm minimization + subspace identification, where nuclear norm
% minimization is done by alternating direction method of multipliers.  The
% model is:
%
% x_t+1 = A*x_t + B*u_t
% y_t ~ Poiss( f( C*x_t + D*u_t + b ) )
%
% All arguments the same as moesp_nn, except for:
% f - known output nonlinearity
% rho - smoothing constant for ADMM
% b - learned output bias (e.g. log average firing rate)
% David Pfau, 2011

addpath /Users/davidpfau/Documents/MATLAB/TFOCS
if nargin < 7
    rho = 1;
end
if nargin < 8
    eps = 1e-3;
end

l = size( y, 1 );
m = size( u, 1 );

U = block_hankel( u, 1, i, N );
assert( size(U,1) < size(U,2) );
[~,~,v] = svd(U);
Un = v(:,m*i+1:end);
s0 = svd( block_hankel( y, 1, i, N ) * Un );

%% Nuclear norm minimization using TFOCS and ADMM, then take svd of YU^\perp
opts = tfocs_SCD;
opts.tol = 1e-4; % don't have all day here, folks...
opts.printEvery = 10;

yh1 = y(:,1:N);
b = log(mean(y,2) + 1e-6);
zyh = zeros(l*(N+1),1); % auxilliary variable for ADMM
res = Inf; % residual
while res > 1e-4 % ADMM loop
    % Nuclear norm minimization
    lambda = 2*s0(1)/l/N/vsig^2;
    yh = tfocs_SCD( smooth_linear( zyh ), ...
        @(varargin) hankel_op( Un, l, i, N, varargin{:} ), ...
        @proj_spectral, ...
        rho, ...
        yh1, ...
        hankel_op( Un, l, i, N, yh1, 1 ), ...
        opts );
    b = b1;
    
    %Newton's method
    yh1 = yh;
    b1  = b;
    eyb = @(yh_,b_) exp( yh_ + b_*ones(1,N) ); % arcane, I know, but this saves a lot of space later
    f = @(yh_,b_) lambda*sum( sum( eyb(yh_,b_) - y(:,1:N).*( yh_ + b_*ones(1,N) ) ) ) ...
        + zyh(:)'*yh_(:) + zb'*b_ + 0.5*rho*( yh_(:)'*yh(:) + b'*b );
    newtres = Inf; % residue for newton step
    while newtres > 1e-12
        eyb1 = eyb(yh1,b1); % saves even more space
        gy = lambda*( eyb1 - y(:,1:N) ) + zyh + rho( yh1 - yh ); % gradient of f wrt yh at (yh,b)
        gb = lambda*( sum( eyb1 - y(:,1:N), 2 ) ) + zb + rho( b1 - b ); % gradient of f wrt b at (yh,b)
        
        % The Hessian is sparse, with many blocks that are diagonal
        % matrices.  We keep track of these diagonals to do fast division
        % by the Hessian.  Let H = [A B; B' C], A is l*N by l*N, C is l*l,
        % both are diagonal, and B is l*N by l and each l*l block is
        % diagonal.  We can then express the inverse Hessian as
        % [ inv(A) + inv(A)*B*inv(C-B'*inv(A)*B)*B'*inv(A),
        %   -inv(A)*B*inv(C-B'*inv(A)*B); -inv(C-B'inv(A)*B)*B'*inv(A),
        %   inv(C-B'*inv(A)*B) ] by the block-matrix inversion lemma and 
        % the Woodbury matrix inversion lemma.  Pardon the arcana below.
        diagA = lambda*eyb1+rho;
        BtinvAgy = ( lambda*gy./diagA )*eyb1';
        CBtinvAB = lambda*sum( eyb1 - eyb1.^2./( eyb1 + rho ) ) + rho; % diagonal of C-B'*inv(A)*B
        dyh = 1./diagA + lambda*diag( ( BtinvAgy - gb )./CBtinvAB )*eyb1./diagA;
        db  = ( gb - BtinvAgy )./CBtinvAB;
        
        yh0 = yh1;       b0  = b1;
        yh1 = yh1 - dyh; b1  = b1  - db;
        
        while f(yh1,b1) > f(yh0,b0)
            yh1 = (yh1 + yh0)/2;
            b1  = (b1  + b0)/2;
        end
        newtres = norm( yh1 - yh0 ) + norm( b1 - b0 );
    end
    
    % Dual variable update
    zyh = zyh + rho*(yh - yh1);
    zb  = zb  + rho*(b - b1);
    res = norm( yh - yh1 ) + norm( b - b1 );
end
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