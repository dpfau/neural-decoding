function A = block_hankel_right_mult_mat(r,s,n,U)
% Returns a sparse matrix that maps a matrix with r rows and n
% columns to a block Hankel matrix with s block-rows, each length r,
% and size(U,1) columns, left-multiplied by U

m = size(U,1);
k = size(U,2);
assert( m + s - 1 <= n, 'Too many rows in output matrix!' );
i = (1:r*s*k)'*ones(1,m);
j = (1:r*s)'*ones(1,k);
j = j(:)*ones(1,m) + ones(r*s*k,1)*(0:r:r*(m-1));
A = sparse(i,j,U(ceil((1:numel(U)*r*s)/(r*s))),r*s*k,r*n);