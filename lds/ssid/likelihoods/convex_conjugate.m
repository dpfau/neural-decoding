function [y z] = convex_conjugate( f, x, z0 )
% Numerically computes the convex conjugate of f at x:
% f*(x) = sup_z <x,z> - f(z)

if nargin == 2
    [~,z0] = f( x );
end
[z,y] = fminunc( @(z) conjugate_objective( f, x, z ), z0, optimset('GradObj','on','Display','iter') );
y = -y;

function [y grad] = conjugate_objective( f, x, z )

[y grad] = f( z );
y = - x(:)'*z(:) + y;
grad = - z + grad;