function [f df] = poiss_loglik( y, k, s, x, t )

if nargin == 3
    error( 'Function does not support proximity operator' )
else
    m = size( y, 1 );
    assert( size( x, 2 ) == size( y, 2 ) + 1 + m * s, 'Parameters do not match the size of the data' );
    if s > 0
        D = x( :, end - m * s + 1 : end );
        my = mean( y, 2 );
        Y = block_hankel( [ my( :, ones( 1, s ) ), y ], 1, s, size( y, 2 ) + s - 1 );
        DY = D*Y;
    else
        DY = 0;
    end
    z = x( :, 1 : end - 1 - m * s ) + x( :, ( end - m * s ) ) * ones( 1, size( x, 2 ) - 1 - m * s ) + DY;
    f = k * sum( sum( -y.*z + exp( z ) ) );
    if nargout == 2
        if s > 0
            df = k * [ -y + exp( z ), sum( -y + exp( z ), 2 ), ( -y + exp( z ) ) * Y' ];
        else
            df = k * [ -y + exp( z ), sum( -y + exp( z ), 2 ) ];
        end
    end
end