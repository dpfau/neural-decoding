function yh = testnucnrm( y, r )

if nargin == 1
    r = 10;
end
m = size( y, 1 );
N = size( y, 2 );
opts = tfocs_SCD;
yh = tfocs_SCD( [], @(x,mode) linearF( m, r, N, x, mode ), @proj_spectral, 1, y, linearF( m, r, N, y, 1 ), opts );

function y = linearF( m, i, N, x, mode )
% Given y, calculates the block-Hankel matrix with i block-rows

switch mode
    case 0
        y = { [ m, N ], [ m * i, N - i + 1 ] };
    case 1
        y = block_hankel( x, 1, i, N );
    case 2
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