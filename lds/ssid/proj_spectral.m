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