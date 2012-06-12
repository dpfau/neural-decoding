function [f df] = poiss_loglik_bias( y, k, e, x, t )
% Poisson log likelihood, including bias term but not history term
% Also includes a small quadratic penalty on the bias, scaled by e

if nargin == 5
    error( 'Function does not support proximity operator' )
else
    assert( size( x, 2 ) == size( y, 2 ) + 1, 'Parameters do not match the size of the data' );
    z = x( :, 1 : end - 1 ) + x( :, end ) * ones( 1, size( x, 2 ) - 1 );
    f = k * sum( sum( -y.*z + exp( z ) ) ) + 1/2 * k * e * x( :, end )' * x( :, end );
    if nargout == 2
        df = k * [ -y + exp( z ), sum( -y + exp( z ), 2 ) + e * x( :, end ) ];
    end
end