function [A C x0 yh s s0] = ssid_poiss_nn( y, i, N, vsig, eps )
addpath /Users/davidpfau/Documents/MATLAB/TFOCS
if nargin < 6
    eps = 1e-3;
end

l = size( y, 1 );
s0 = svd( block_hankel( y, 1, i, N ) );

%% Nuclear norm minimization using TFOCS, then take svd of Y
opts = tfocs_SCD;
opts.tol = 1e-3; % don't have all day here, folks...
opts.printEvery = 1;
yh = tfocs_SCD( @(varargin) poiss_like( y(:,1:N), s0(1)/l/N/vsig^2, varargin{:} ), ...
    @(x,mode) hankel_op( l, i, N, x, mode ), ...
    @proj_spectral, ...
    1e-12, ...
    y(:,1:N), ...
    hankel_op( l, i, N, y(:,1:N), 1 ), ...
    opts );
[r,s,v] = svd( hankel_op( l, i, N, yh, 1 ) );

%% Approximate order of the system
n = find( diag( s )/s(1) < eps, 1 ) - 1; 
if isempty( n ), n = 10; end

%% Recover A, C, x0
G = r( :, 1:n ) * sqrt( s( 1:n, 1:n ) );
C = G( 1:l, : );
A = pinv( G( 1:(i-1)*l, : ) ) * G( (l+1):i*l, : );
A = stabilize( A );
x0 = v( 1, 1:n ) * sqrt( s( 1:n, 1:n ) );

function y = hankel_op( m, i, N, x, mode )
% Given x, calculates X*Un, where X is a block-Hankel matrix derived from
% x, and Un is a matrix whose columns span the null space of the
% block-Hankel matrix of u.  Standard form of linear operators in TFOCS.

switch mode
    case 0 % return { input size, output size }
        y = { [ m, N ], [ m * i, N - i + 1 ] };
    case 1 % apply operator to input
        y = block_hankel( x, 1, i, N );
    case 2 % apply adjoint operator to input
        y = adjoint_hankel( x, i, N );
end

function [hx, x] = proj_spectral( x, t )
% find the projection of the matrix x onto the spectral norm ball, that is,
% set all singular values greater than 1 to 1.  t is included to meet the
% form of a generalized projection function for TFOCS.

hx = 0;
if nargin == 2
    [u,s,v] = svd( x );
    x = u * min( s, 1 ) * v';
elseif nargout == 2
    error( 'This function is not differentiable.' );
end
s = svd( x );
if s( 1 ) > 1, hx = Inf; end

function [hx, x] = poiss_like( y, c, x, t )
% Log Likelihood of data y given rates x in prox-capable form for TFOCS
% When passing to TFOCS with specific data y, pass @(x,t) poiss_like(y,x,t)

if nargin == 4
    x = 0.5*( x+c*t + sqrt( ( x+c*t ).^2 - 4*c*t*y ) );
end
hx = c*( sum( log( x( y ~= 0 ) ).*y( y ~= 0 ) ) - sum( sum( x + gammaln( y+1 ) ) ) );
if nargin ~= 4 && nargout == 2
    x = c*( y./x - 1 );% in this case the second output is the gradient of hx wrt x
end