function [A B C D x0 s] = moesp( y, u, i, N, opts )
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

if ~isfield( opts, 'noise' ),       opts.noise = 'none';    end
if ~isfield( opts, 'proj' ),        opts.proj = 'orth_svd'; end
if ~isfield( opts, 'tol' ),         opts.tol = 1e-3;        end
if ~isfield( opts, 'maxOrder' ),    opts.maxOrder = 10;     end
if ~isfield( opts, 'instant' ),     opts.instant = 1;       end
if ~strcmpi( opts.noise, 'none' )
    if ~isfield( opts, 'vsig' ),    opts.vsig = 1;          end
    if strcmpi( opts.noise, 'poiss' )
        if ~isfield( opts, 'rho' )  opts.rho = 1;           end
    end
end
if ~isfield( opts, 'tfocs_path' )
    opts.tfocs_path = '/Users/davidpfau/Documents/MATLAB/TFOCS'; 
end

%% Project the columns of Y or Yf onto the appropriate subspace
if strcmpi( opts.proj, 'oblique' )
    assert( ~strcmp( opts.noise, 'none' ), 'Cannot use oblique projection with nuclear norm minimization' )
    Y = block_hankel( y, 1, 2*i, N );
    U = block_hankel( u, 1, 2*i, N );
    
    Yf = Y(i+1:end,:);
    Uf = U(i+1:end,:);
    
    Oi = oblique( Yf, Uf, [Y(1:i,:);U(1:i,:)] );
elseif strncmpi( opts.proj, 'orth', 4 )
    Y = block_hankel( y, 1, i, N );
    U = block_hankel( u, 1, i, N );
    
    if strcmpi( opts.proj, 'orth_pinv' )
        Un = eye( N - i + 1 ) - pinv( U ) * U;
    elseif strcmpi( opts.proj, 'orth_svd' )
        [~,~,v] = svd(U);
        Un = v(:,m*i+1:end);
    end
    
%% If appropriate, compensate for output noise with nuclear norm minimization
    switch opts.noise
        case 'none'
            Oi = Y*Un;
        case 'gauss'
            % For Gaussian noise, can do nuclear norm plus Gaussian 
            % likelihood minimization directly
            addpath( opts.tfocs_path )
            tfocs_opts = tfocs_SCD;
            tfocs_opts.tol = 1e-4; % don't have all day here, folks...
            tfocs_opts.printEvery = 10;
            yh = tfocs_SCD( [], ...
                @(varargin) hankel_op( Un, l, i, N, varargin{:} ), ...
                @proj_spectral, ...
                2*s0(1)/l/N/opts.vsig^2, ...
                y(:,1:N), ...
                hankel_op( Un, l, i, N, y(:,1:N), 1 ), ...
                tfocs_opts );
            Oi = hankel_op( Un, l, i, N, yh, 1 );
        case 'poiss'
            % For Poisson noise, use ADMM to optimize jointly over nuclear
            % norm and Poisson likelihood
            addpath( opts.tfocs_path )
            tfocs_opts = tfocs_SCD;
            tfocs_opts.tol = 1e-4;
            tfocs_opts.maxIts = 1e3;
            tfocs_opts.printEvery = 0;
            
            yh1 = y(:,1:N);
            b1 = log(mean(y,2) + 1e-6);
            zyh = zeros(size(yh1)); % auxilliary variable for ADMM
            zb  = zeros(size(b1));
            r_norm = Inf; % residual
            lambda = 2*s0(1)/l/N/opts.vsig^2; % tradeoff between nuclear norm and output log likelihood
            
            fprintf('%3s\t%10s\t%10s\t%10s\n', 'iter', ...
                'r norm', 's norm', 'objective');
            iter = 0;
            while r_norm > 1e-4 % ADMM loop
                iter = iter + 1;
                % Nuclear norm minimization
                yh = tfocs_SCD( [], ...
                    @(varargin) hankel_op( Un, l, i, N, varargin{:} ), ...
                    @proj_spectral, ...
                    opts.rho, ...
                    yh1 - zyh, ...
                    hankel_op( Un, l, i, N, yh1, 1 ), ...
                    tfocs_opts );
                b = b1;
                
                % Minimize augmented negative log likelihood
                opts1 = struct( 'GradObj', 'on', ...
                    'LargeScale', 'on', ...
                    'Hessian', 'on', ...
                    'HessMult', @hessmult, ...
                    'Display', 'off' );
                xold = [yh1, b1];
                x1 = fminunc( @(x) f( x, [yh,b], lambda, opts.rho, y(:,1:N), [zyh,zb] ), [yh,b], opts1 );
                yh1 = x1(:,1:end-1);
                b1  = x1(:,end);
                
                % Dual variable update
                zyh = zyh + yh - yh1;
                zb  = zb  + b - b1;
                r_norm = norm( yh - yh1, 'fro' ) + norm( b - b1, 'fro' );
                s_norm = norm( -opts.rho*( [yh1, b1] - xold ), 'fro' );
                obj = objective( yh, yh1, b1, y(:,1:N), lambda, i, Un );
                fprintf('%3d\t%10.4f\t%10.4f\t%10.2f\n', iter, r_norm, s_norm, obj );
            end
            Oi = hankel_op( Un, l, i, N, yh, 1 );
        otherwise
            error(['''' opts.noise ''' is not a recognized output noise.']);
    end
else
    error(['''' opts.proj ''' is not a recognized projection method.']);
end

%% Reconstruct A, C
[r,s,~] = svd( Oi );
n = find( diag( s )/s(1) < opts.tol, 1 ) - 1; % approximate order of the system
if isempty( n )
    n = opts.maxOrder;
else
    n = min( n, opts.maxOrder );
end

G = r( :, 1:n ) * sqrt( s( 1:n, 1:n ) );
C = G( 1:l, : );
A = pinv( G( 1:(i-1)*l, : ) ) * G( (l+1):i*l, : );

%% Stabilize A
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
else
    xx = pinv( [F1 F3], 1e-6 ) * y( 1:l*N )';
    x0 = xx(1:n);
    B = reshape( xx(n+1:end), n, m );
    D = zeros(l,m);
end
s = diag(s);

function [y grad Hinfo] = f(x,x1,lam,rho,dat,z)
% Objective function/gradient/fields necessary for multiplication by the
% Hessian for minimizing the log likelihood term in ADMM

N = size(x,2) - 1;
yh = x(:,1:N);
b = x(:,end);
eyb = exp( yh + b*ones(1,N) );

y = lam*sum( sum( eyb - dat.*( yh + b*ones(1,N) ) ) ) ...
    + 0.5*rho*( (x1(:) - x(:) + z(:))'*(x1(:) - x(:) + z(:)) );
grad = lam*( [eyb - dat, sum( eyb - dat, 2 )] ) + rho*( x - x1 - z );
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

function obj = objective(yh,yh1,b1,y,lam,i,Un)

l = size(yh,1);
N = size(yh,2);
obj = sum( svd( hankel_op( Un, l, i, N, yh, 1 ) ) ) ...
    + lam*sum( sum( exp( yh1 + b1*ones(1,N) ) - y.*( yh1 + b1*ones(1,N) ) ) );