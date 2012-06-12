function [f df] = poiss_loglik_bias_conj( y, k, e, x, t )

if nargin == 5
    error( 'Function does not support proximity operator' )
else
    z = newton_root( @(
    [fz gradz] = poiss_loglik_bias( y, k, e, z );
    f = z(:)'*gradz - fz;
end