function [f df Hf] = poiss_loglik_bias( y, k, e, x, t )
% Poisson log likelihood, including bias term but not history term
% Also includes a small quadratic penalty on the bias, scaled by e

if nargin == 5
    error( 'Function does not support proximity operator' )
else
    if isempty( x )
        f = zeros( size( y, 1 ), size( y, 2 ) + 1 );
    else
        assert( size( x, 2 ) == size( y, 2 ) + 1, 'Parameters do not match the size of the data' );
        m = size( x, 1 );
        n = size( x, 2 );
        z = x( :, 1 : end - 1 ) + x( :, end ) * ones( 1, n - 1 );
        ez = exp( z );
        f = k * sum( sum( -y.*z + ez ) ) + 1/2 * k * e * x( :, end )' * x( :, end );
        if nargout >= 2
            df = k * [ -y + ez, sum( -y + ez, 2 ) + e * x( :, end ) ];
            if nargout == 3
                i = [ 1 : m * ( n - 1 ), 1 : m * ( n - 1 ), repmat( m * ( n - 1 ) + ( 1 : m ), 1, n ) ] ;
                j = [ 1 : m * ( n - 1 ), repmat( m * ( n - 1 ) + ( 1 : m ), 1, n - 1 ), 1 : m * ( n - 1 ), m * ( n - 1 ) + ( 1 : m ) ];
                h = k * [ ez, ez, ez, e + sum( ez, 2 ) ];
                Hf = sparse( i, j, h, numel( x ), numel( x ) );
            end
        end
    end
end