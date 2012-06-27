function [f df Hf] = poiss_loglik_bias_history( y, k, e, g, s, x, t )
% Poisson log likelihood, including bias and history term, as well as
% isotropic Gaussian prior on the bias and the history terms.

if nargin == 7
    if numel( t ) == 1
        error( 'Function does not support proximity operator' )
    else
        f = 1;
    end
else
    m = size( y, 1 );
    n = size( y, 2 );
    if isempty( x )
        f = zeros( m, n + 1 + m * s );
    else
        assert( ( size( x, 2 ) == n + 1 + m * s ) & ( size( x, 1 ) == m ), 'Parameters do not match the size of the data' );
        if s > 0
            D = x( :, end - m * s + 1 : end );
            my = mean( y, 2 );
            Y = block_hankel( [ my( :, ones( 1, s ) ), y ], 1, s, n + s - 1 );
            DY = D*Y;
        else
            D  = 0;
            DY = 0;
        end
        z = x( :, 1 : n ) + x( :, n + 1 ) * ones( 1, n ) + DY;
        ez = exp( z );
        f = k * sum( sum( -y.*z + ez ) ) ...
            + 1/2 * k * e * x( :, n + 1 )' * x( :, n + 1 ) ...
            + 1/2 * k * g * D( : )' * D( : );
        if nargout >= 2
            if s > 0
                df = k * [ -y + ez, sum( -y + ez, 2 ) + e * x( :, n + 1 ), ( -y + ez ) * Y' + g * D ];
            else
                df = k * [ -y + ez, sum( -y + ez, 2 ) + e * x( :, n + 1 ) ];
            end
            if nargout == 3
                if s > 0
                    ezY = tprod( ez, [1 2], Y, [3 2] );
                    xx = 1 : m * n; % Diagonal indices are same for i and j
                    xb = repmat( m * n + ( 1 : m ), 1, n );
                    bb = m * n + ( 1 : m );
                    xd = repmat( reshape( m * ( n + 1 ) + ( 1 : m * m * s ), m, m * s  ), n, 1 );
                    bd = m * ( n + 1 ) + ( 1 : m * m * s );
                    dd_i = repmat( bd, 1, m * s );
                    dd_j = repmat( reshape( bd, m, m * s ), m * s, 1 );
                    i = [ xx, xx, xb, bb, repmat( xx, 1, m * s ), xd( : )', repmat( bb, 1, m * s ), bd, dd_i ];
                    j = [ xx, xb, xx, bb, xd( : )', repmat( xx, 1, m * s ), bd, repmat( bb, 1, m * s ), dd_j( : )' ];
                    h = k * [ ez, ez, ez, e + sum( ez, 2 ), ...
                         reshape( ezY, m, n * m * s ), reshape( ezY, m, n * m * s ), squeeze( sum( ezY, 2 ) ), squeeze( sum( ezY, 2 ) ), ...
                         reshape( tprod( tprod( ez, [1 3], Y, [2 3] ), [1 2 -1], Y, [3 -1] ), m, m * m * s * s ) ];
                else
                    i = [ 1 : m * n, 1 : m * n, repmat( m * n + ( 1 : m ), 1, n + 1 ) ] ;
                    j = [ 1 : m * n, repmat( m * n + ( 1 : m ), 1, n ), 1 : m * n, m * n + ( 1 : m ) ];
                    h = k * [ ez, ez, ez, e + sum( ez, 2 ) ];
                end
                Hf = sparse( i, j, h, numel( x ), numel( x ) );
                if s > 0
                    Hf = Hf + g * k * sparse( m * ( n + 1 ) + ( 1 : m * m * s ), m * ( n + 1 ) + ( 1 : m * m * s ), ones( 1, m * m * s ) );
                end
            end
        end
    end
end