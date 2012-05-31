function [f df] = poiss_loglik_conj( y, k, s, x, t )

if nargin == 3
    error( 'Function does not support proximity operator' )
else
    f = sum( sum( ( log( 1/k * x + y ) - 1 ) .* ( x + k * y ) ) );
    if nargout == 2
        df = log( 1/k * x + y );
    end
end