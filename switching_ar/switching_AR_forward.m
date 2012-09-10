function [z c l] = switching_AR_forward( x, A, Q, T, p0, mx )

tiny = 1e-300;
if nargin < 6
    mx = zeros( size( x, 1 ), 1 );
end
x = x - mx( :, ones( 1, size( x, 2 ) ) );
n = length( A );
m = size( x, 1 );
k = size( A{1}, 2 ) / m;
t = size( x, 2 );
z = zeros( n, t-k );
c = zeros( 1, t-k );

x_ = zeros( m*k, t-k );
for i = 1:k
    x_((1:m) + m*(i-1),:) = x(:,i:end-k+i-1);
end

l = zeros( n, t-k );
lnorm = zeros( n ); % normalizing factor
r = cell( n, 1 );
for i = 1:n
    lnorm(i) = -m/2*log(2*pi) - sum(log(diag(chol(Q{i}))));
    r{i} = x(:,k+1:end) - A{i} * x_;
    Qinv = Q{i}^-1;
    l(i,:) = exp( lnorm(i) - 1/2*tprod( tprod( Qinv, [-1 1], r{i}, [-1 2], 'n'), [-1 2], r{i}, [-1 2], 'n' ) );
end

zt = p0;
for i = 1:t-k
    zt = diag(l(:,i))*T*zt;
    c(i) = sum(zt);
    zt = zt/c(i);
    if c(i) == 0 % In case of numerical underflow, just reset everything
        c(i) = tiny;
        zt = p0;
    end
    z(:,i) = zt;
end