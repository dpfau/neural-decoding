function x = find_pos_rand( A, tol )
N = size( A, 2 );
x = randn( N, 1 );
n = nnz( A*x <= 0 );
while n > 0
    idx = rand( N, 1 ) > tol/N;
    x_old = x;
    x( idx ) = randn( nnz( idx ), 1 );
    n_ = nnz( A*x <= 0 );
    if n_ < n %|| rand < tol^(n_-n)
        if n ~= n_, fprintf('%d negative entries\n', n); end
        n = n_;
    else
        x = x_old;
    end
end