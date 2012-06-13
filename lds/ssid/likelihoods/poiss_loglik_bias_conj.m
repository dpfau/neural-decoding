function [f df] = poiss_loglik_bias_conj( y, k, e, x, t )

if nargin == 5
    error( 'Function does not support proximity operator' )
else
    z = newton_root( @(z) poiss_loglik_bias_jacobian( y, k, e, z ), x );
    [fz gradz] = poiss_loglik_bias( y, k, e, z );
    f = z(:)'*gradz(:) - fz;
end
df = 0;

function [fz Jz] = poiss_loglik_bias_jacobian( y, k, e, x )
% Returns the gradient and the Jacobian of the gradient (that is, the
% Hessian) of the log likeilihood of the poisson GLM with bias term, for
% use in computing the x for which the gradient of poiss_loglik_bias = z,
% needed to compute the convex conjugate of poiss_loglik_bias

m = size( x, 1 );
n = size( x, 2 );
z = x( :, 1 : end - 1 ) + x( :, end ) * ones( 1, n - 1 );
ez = exp( z );
fz = k * [ -y + ez, sum( -y + ez, 2 ) + e * x( :, end ) ];
i = [ 1 : m * ( n - 1 ), 1 : m * ( n - 1 ), repmat( m * ( n - 1 ) + ( 1 : m ), 1, n ) ] ;
j = [ 1 : m * ( n - 1 ), repmat( m * ( n - 1 ) + ( 1 : m ), 1, n - 1 ), 1 : m * ( n - 1 ), m * ( n - 1 ) + ( 1 : m ) ];
h = k*[ ez, ez, ez, e + sum( ez, 2 ) ];
Jz = sparse( i, j, h, numel( x ), numel( x ) );