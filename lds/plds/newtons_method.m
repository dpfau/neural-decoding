function [x fx Hinfo] = newtons_method( f, x0, eps )

if nargin == 2
    eps = 1e-6;
end

fx0 = Inf;
[fx,grad,Hinfo]  = f(x0);

i = 0;
while fx0 - fx > eps
    i = i+1;
    fx0 = fx;
    
    if issparse( Hinfo )
        H = Hinfo;
    elseif isstruct( Hinfo )
        H = sparse_hess( Hinfo );
    end
    
    x = reshape( x0(:) - H\grad(:), size(x0) );
    while f(x) >= fx0 % If we overshoot the minimum
        x = x0 + (x - x0)/2;
    end
    [fx,grad,Hinfo] = f(x);
    fprintf('Iter: %i, f(x) = %d\n',i,fx);
    x0 = x;
end