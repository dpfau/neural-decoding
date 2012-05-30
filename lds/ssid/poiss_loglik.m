function [f df] = poiss_loglik( y, s, x, t )

if nargin == 3
    error( 'Function does not support proximity operator' )
else
    z = x(:,1:end-1) + x(:,end*ones(1,size(x,2)-1));
    f = s*sum( sum( -y.*z + exp( z ) ) );
    if nargout == 2
        df = s*[ -y + exp( z ), sum( -y + exp( z ), 2 )];
    end
end