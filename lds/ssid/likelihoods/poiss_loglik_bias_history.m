function [f df Hf] = poiss_loglik_bias_history( y, k, e, g, s, x, t )
% Poisson log likelihood, including bias and history term, as well as
% isotropic Gaussian prior on the bias and the history terms.

if nargin == 5
    error( 'Function does not support proximity operator' )
else
    m = size( y, 1 );
    n = size( y, 2 );
    if isempty( x )
        f = zeros( m, n + 1 + m * s );
    else
        assert( size( x, 2 ) == n + 1 + m * s, 'Parameters do not match the size of the data' );
        if s > 0
            D = x( :, end - m * s + 1 : end );
            my = mean( y, 2 );
            Y = block_hankel( [ my( :, ones( 1, s ) ), y ], 1, s, n + s - 1 );
            DY = D*Y;
        else
            DY = 0;
        end
        z = x( :, 1 : n ) + x( :, n + 1 ) * ones( 1, n ) + DY;
        ez = exp( z );
        f = k * sum( sum( -y.*z + ez ) ) ...
            + 1/2 * k * e * x( :, n + 1 )' * x( :, n + 1 ) ...
            + 1/2 * k * g * x( :, n + 2 : end )' * x( :, n + 2 : end );
        if nargout >= 2
            if s > 0
                df = k * [ -y + ez, sum( -y + ez, 2 ) + e * x( :, n + 1 ), ( -y + ez ) * Y' + g * x( :, n + 2 : end ) ];
            else
                df = k * [ -y + ez, sum( -y + ez, 2 ) + e * x( :, n + 1 ) ];
            end
            if nargout == 3
                if s > 0
                    i = [ 1 : m * ( n - 1 ), 1 : m * ( n - 1 ), repmat( m * ( n - 1 ) + ( 1 : m ), 1, n ) ] ;
                    j = [ 1 : m * ( n - 1 ), repmat( m * ( n - 1 ) + ( 1 : m ), 1, n - 1 ), 1 : m * ( n - 1 ), m * ( n - 1 ) + ( 1 : m ) ];
                    h = k * [ ez, ez, ez, e + sum( ez, 2 ) ];
                else
                    i = [ 1 : m * ( n - 1 ), 1 : m * ( n - 1 ), repmat( m * ( n - 1 ) + ( 1 : m ), 1, n ) ] ;
                    j = [ 1 : m * ( n - 1 ), repmat( m * ( n - 1 ) + ( 1 : m ), 1, n - 1 ), 1 : m * ( n - 1 ), m * ( n - 1 ) + ( 1 : m ) ];
                    h = k * [ ez, ez, ez, e + sum( ez, 2 ) ];
                end
                Hf = sparse( i, j, h, numel( x ), numel( x ) );
            end
        end
    end
end