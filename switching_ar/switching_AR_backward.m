function [z_ V] = switching_AR_backward( x, A, T, z, c, l )

tiny = 1e-300;
n = length( A );
m = size( x, 1 );
k = size( A{1}, 2 ) / m;
t = size( x, 2 );
z_ = ones( n, t-k );
V = zeros( n );
for i = t-k-1:-1:1
    z_(:,i) = z_(:,i+1)'*diag(l(:,i+1))*T/c(i+1);
    if c(i+1) == tiny
        z_(:,i) = ones( n, 1 );
    end
    V = V + ( (z_(:,i+1)*z(:,i)') .* (diag(l(:,i+1))*T) )/c(i+1);
end
z_ = z.*z_;