function f_conj = make_convex_conjugate( f, z0 )
% Returns function handle that numerically computes convex conjugate via Newton's method:
% f*(x) = sup_z <x,z> - f(z)

    if nargin == 1
        z0 = f( [] );
    end
    
    function [y grad hess] = conjugate_objective( f, x, z )
        
        [y grad hess] = f( z );
        y = - x(:)'*z(:) + y;
        grad = - x + grad;
        
    end

    function [y grad] = convex_conjugate( f, x, t )
        % TFOCS compatible calling convention
        
        if nargin == 3
            error( 'Function does not support proximity operator' )
        end
        [grad,y] = newtons_method( @(z) conjugate_objective( f, x, z ), z0, 1e-6, 0 );
        y = -y;
        z0 = grad; % Closure!
    end

f_conj = @(varargin) convex_conjugate( f, varargin{:} );

end