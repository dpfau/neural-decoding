function f_conj = make_convex_conjugate( f )
% Returns function handle that numerically computes convex conjugate via Newton's method:
% f*(x) = sup_z <x,z> - f(z)

f_conj = @(varargin) convex_conjugate( f, varargin{:} );

function [y grad hess] = conjugate_objective( f, x, z )

[y grad hess] = f( z );
y = - x(:)'*z(:) + y;
grad = - x + grad;

function y = convex_conjugate( f, x, z0 )

if nargin == 2
    z0 = f( [] );
end
[~,y] = newtons_method( @(z) conjugate_objective( f, x, z ), z0, 1e-6, 1 );
y = -y;