function [f df] = poiss_loglik( y, k, x, t )
% Poisson log likelihood, without bias or history term

if nargin == 4
    error( 'Function does not support proximity operator' )
else
    assert( size( x, 2 ) == size( y, 2 ), 'Parameters do not match the size of the data' );
    f = k * sum( sum( -y.*x + exp( x ) ) );
    if nargout == 2
        df = k * ( -y + exp( x ) );
    end
end