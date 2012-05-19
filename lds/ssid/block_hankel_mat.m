function A = block_hankel_mat(a,b)
% Returns a sparse matrix that maps a matrix with a(1) rows and a(2)
% columns to a block Hankel matrix with b(1) block-rows, each length a(1),
% and b(2) columns, where b(2) <= a(1)-b(1)+1

assert( b(2) + b(1) - 1 <= a(2), 'Too many rows in output matrix!' );
k = a(1)*prod(b);
j = (1:(a(1)*b(1)))'*ones(1,b(2)) + a(1)*ones(a(1)*b(1),1)*(0:b(2)-1);
A = sparse(1:k,j(:),ones(1,k),k,prod(a));