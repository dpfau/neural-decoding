function [spec s0] = test_nn_spec_opt( y, u, i, vsigs )

addpath /Users/davidpfau/Documents/MATLAB/TFOCS

N = size(y,2);

l = size( y, 1 );
m = size( u, 1 );
spec = zeros( size(y,1)*i, length(vsigs) );

U = block_hankel( u, 1, i, N );
assert( size(U,1) < size(U,2) );
[~,~,v] = svd(U);
Un = v(:,m*i+1:end);
s0 = svd( block_hankel( y, 1, i, N ) * Un );

opts = tfocs_SCD;
opts.tol = 1e-4; % don't have all day here, folks...
opts.printEvery = 10;
for t = 1:length(vsigs)
    yh = tfocs_SCD( [], ...
        @(varargin) hankel_op( Un, l, i, N, varargin{:} ), ...
        @proj_spectral, ...
        2*s0(1)/l/N/vsigs(t)^2, ...
        y(:,1:N), ...
        hankel_op( Un, l, i, N, y(:,1:N), 1 ), ...
        opts );
    spec(:,t) = svd( hankel_op( Un, l, i, N, yh, 1 ) );
end