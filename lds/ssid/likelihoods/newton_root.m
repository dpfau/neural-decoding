function z = newton_root( f, x, z0, eps )
% Solves via Newton's method the nonlinear system of equations
% f( z ) = x
% where f is a vector-valued function which returns its Jacobian at that
% point

if nargin < 4
    eps = 1e-5;
end
if nargin < 3
    z0 = zeros( size( x ) );
end

z = z0;
[fz Jz] = f( z );
while norm( fz - x ) > eps
    % Solve Jz*dz = -fz by stabilized biconjugate gradient
    dz = zeros( numel( z ), 1 );
    r  = -fz(:) - Jz*dz;
    r_ = r;
    p  = r;
    p_ = r_;
    while norm( Jz*dz + fz ) > eps
        a = ( r_'*r )/( p_'*Jz*p );
        
        r1  = r  - a*Jz*p;
        r_1 = r_ - a*Jz'*p_;
        
        b = ( r1'*r_1 )/( r'*r_ );
        
        p  = r1  + b*p;
        p_ = r_1 + b*p_;
        
        r  = r1;
        r_ = r_1;

        dz = dz + a*p;
    end
    z = z + reshape( dz, size( z ) );
    [fz Jz] = f( z );
end