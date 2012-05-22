function [hx x] = proj_nuclear_hankel( Un, m, i, N, z, t )
% The prox-capable version of the function 
% || block_hankel( z, 1, i, N )*Un ||_*, where ||.||_* is the sum of
% singular values or nuclear norm.  Solve the generalized projection
% problem by running *another* tfocs program

nn_hankel = @(y) sum( svd( block_hankel( y, 1, i, N ) * Un ) );
if nargin == 5
    hx = nn_hankel(z);
    if nargout == 2
        error( 'Nuclear norm is not differentiable.' )
    end
else
    opts = tfocs_SCD;
    opts.tol = 1e-4;
    opts.maxIts = 1e3;
    opts.printEvery = 1;
    x = tfocs_SCD([],...
                  @(varargin) hankel_op( Un, m, i, N, varargin{:} ), ...
                  @proj_spectral,...
                  1/t,...
                  z,...
                  hankel_op( Un, m, i, N, z, 1 ),...
                  opts);
    hx = nn_hankel(x) + 1/(2*t)*(x(:)-z(:))'*(x(:)-z(:));
end