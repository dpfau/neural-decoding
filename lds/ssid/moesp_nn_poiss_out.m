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
    
    % Minimize augmented negative log likelihood
    opts1 = struct( 'GradObj', 'on', ...
               'LargeScale', 'on', ...
               'Hessian', 'on', ...
               'HessMult', @hessmult, ...
               'Display', 'iter' );
    x1 = fminunc( @(x) f(x,[yh,b],lambda,rho,y(:,1:N),[zyh,zb]), [yh,b], opts1 );
    yh1 = x1(:,1:end-1);
    b1  = x1(:,end);
    
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

function [y grad Hinfo] = f(x,x1,lam,rho,dat,z)
% Objective function/gradient/fields necessary for multiplication by the
% Hessian for minimizing the log likelihood term in ADMM

N = size(x,2) - 1;
yh = x(:,1:N);
b = x(:,end);
eyb = exp( yh + b*ones(1,N) );

y = lam*sum( sum( eyb - dat.*( yh + b*ones(1,N) ) ) ) ...
    + z(:)'*x(:) + 0.5*rho*( (x(:) - x1(:))'*(x(:) - x1(:)) );
grad = lam*( [eyb - dat, sum( eyb - dat, 2 )] ) + z + rho*( x - x1 );
Hinfo = struct('lam',lam,'rho',rho,'eyb',eyb);

function hv = hessmult(Hinfo,v)
% Multiplication by the Hessian of the objective for the log likelihood
% step in ADMM

N = size(Hinfo.eyb,2);
hv = zeros(size(v));
for i = 1:size(v,2)
    v2 = reshape(v(:,i),size(Hinfo.eyb,1),N+1);
    hv2 = [ (Hinfo.lam*Hinfo.eyb + Hinfo.rho).*v2(:,1:N) + Hinfo.lam*Hinfo.eyb.*(v2(:,end)*ones(1,N)), ...
            Hinfo.lam*sum(Hinfo.eyb.*v2(:,1:N),2) + (Hinfo.lam*sum(Hinfo.eyb,2) + Hinfo.rho).*v2(:,end) ];
    hv(:,i) = hv2(:);
end