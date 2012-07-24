function [f df] = poiss_loglik_conj( y, k, x, t )

if nargin == 4
    error( 'Function does not support proximity operator' )
else
    if nnz( 1/k * x + y < 0 ) > 0
        f = +Inf;
        if nargout == 2
            df = zeros( size( x ) );
        end
    else
        idx = 1/k * x + y ~= 0; % Handles edge cases
        f = sum( sum( ( log( 1/k * x(idx) + y(idx) ) - 1 ) .* ( x(idx) + k * y(idx) ) ) );
        if nargout == 2
            df = log( 1/k * x + y );
        end
    end
end