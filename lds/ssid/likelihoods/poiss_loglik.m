function [f df Hf] = poiss_loglik( y, k, x, t )
% Poisson log likelihood, without bias or history term

if nargin == 4
    % Overload the 4th argument.  One possible use is in evaluating the
    % proximity operator for TFOCS, in which case throw an error.  Another
    % possible use is to evaluate whether the particular vector passed is
    % in the domain of the convex conjugate.
    if numel( t ) == 1
        error( 'Function does not support proximity operator' )
    else
        f = nnz( 1/k * t + y < 0 ) == 0;
    end
else
    if isempty( x ) % Used for computing the convex conjugate.  Returns vector in the domain of the function to initialize a minimization
        f = zeros( size( y ) );
    else
        assert( size( x, 2 ) == size( y, 2 ), 'Parameters do not match the size of the data' );
        ex = exp( x );
        f = k * sum( sum( -y.*x + ex ) );
        if nargout >= 2
            df = k * ( -y + ex );
            if nargout == 3
                Hf = sparse( 1:numel( x ), 1:numel( x ), k * ex );
            end
        end
    end
end