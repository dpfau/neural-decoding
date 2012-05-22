function [f g] = prox_quad( P, q, r, x, t )
% Constructs a function f(x) = 1/2<x,P*x> + <q,x> + r that is in the format
% of a prox-capable function in TFOCS, that is, in addition to being able
% to evaluate the function itself and its gradient at the current point,
% must be able to evaluate Phi_f(z,t) = armin_x f(x) + 1/t*2||x-z||^2

assert( size(P,1) == size(P,2) || size(P,2) == 1, 'P must be square' )
assert( size(q,1) == size(x,1) && size(q,2) == size(x,2), 'q must be same size as x' )
assert( numel(r) == 1, 'r must be scalar' )
if nargin == 4
    f = h( x, P, q, r );
    if nargout == 2
        if numel(P) == 1
            g = P*x + q;
        elseif size(P,2) == 1
            g = reshape(P.*x(:),size(x)) + q;
        else
            g = reshape(P*x(:),size(x)) + q;
        end
    end
else
    assert( numel(t) == 1, 't must be scalar' );
    if numel(P) == 1
        g = (x/t - q)/(P - 1/t);
    elseif size(P,2) == 1
        g = (x/t - q)./(P - 1/t*ones(size(P)));
    else
        g = (P + eye(numel(x))/t)\(x/t - q);
    end
    f = h( g, P, q, r );
end

function y = h( x, P, q, r )

if numel(P) == 1
    y = P/2*(x(:)'*x(:)) + q(:)'*x(:) + r;
elseif size(P,2) == 1
    y = 1/2*x(:)'*(P.*x(:)) + q(:)'*x(:) + r;
else
    y = 1/2*x(:)'*P*x(:) + q(:)'*x(:) + r;
end