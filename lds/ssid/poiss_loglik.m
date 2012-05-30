function [f df] = poiss_loglik( y, k, s, x, t )

if nargin == 3
    error( 'Function does not support proximity operator' )
else
    m = size( y, 1 );
    D = x( :, end - m * s + 1 : end );
    z = x( :, 1 : end - 1 - m * s ) + x( :, ( end - m * s ) * ones( 1, size( x, 2 ) - 1 ) );
    f = k * sum( sum( -y.*z + exp( z ) ) );
    if nargout == 2
        df = k * [ -y + exp( z ), sum( -y + exp( z ), 2 ) ];
    end
end