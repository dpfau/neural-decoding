function [A Q T p0] = switching_AR_M_step( x, z, V, k )

n = size( z, 1 );
m = size( x, 1 );
t = size( x, 2 );
A = cell( n, 1 );
Q = cell( n, 1 );

x_ = zeros( m*k, t-k );
for i = 1:k
    x_((1:m) + m*(i-1),:) = x(:,i:end-k+i-1);
end

for i = 1:n
    A{i} = tprod( z(i,:)', -1, tprod( x(:,k+1:end), [1 3], x_, [2 3] ), [1 2 -1], 'n' ) ...
                 * pinv( tprod( z(i,:)', -1, tprod( x_, [1 3], x_, [2 3] ), [1 2 -1], 'n' ) );
    Q{i} =     ( tprod( z(i,:)', -1, tprod( x(:,k+1:end), [1 3], x(:,k+1:end), [2 3] ), [1 2 -1], 'n' ) ...
        - A{i} * tprod( z(i,:)', -1, tprod( x(:,k+1:end), [2 3], x_, [1 3] ), [1 2 -1], 'n' ) + 0.1*k*t*eye(m) ) ...
        / ( 0.1*k*t + sum( z(i,:) ) );
end
T = V./(ones(n,1)*sum(V));
p0 = z(:,1);