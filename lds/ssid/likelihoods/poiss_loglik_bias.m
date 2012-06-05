function [f df] = poiss_loglik_bias( y, k, x, t )
% Poisson log likelihood, including bias term but not history term

if nargin == 4
    error( 'Function does not support proximity operator' )
else
    assert( size( x, 2 ) == size( y, 2 ) + 1, 'Parameters do not match the size of the data' );
    z = x( :, 1 : end - 1 ) + x( :, end ) * ones( 1, size( x, 2 ) - 1 );
    f = k * sum( sum( -y.*z + exp( z ) ) );
    if nargout == 2
        df = k * [ -y + exp( z ), sum( -y + exp( z ), 2 ) ];
    end
end