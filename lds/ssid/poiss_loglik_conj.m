function [f df] = poiss_loglik_conj( y, k, s, x, t )

if nargin == 3
    error( 'Function does not support proximity operator' )
else
end