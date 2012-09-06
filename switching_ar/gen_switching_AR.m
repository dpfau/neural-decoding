function dat = gen_switching_AR( A, Q, T, p0, mx, t )

n = length(A);
m = size(A{1},1);
k = floor(size(A{1},2)/m);
dat = zeros( m, t + k );
drand = @(p) find( rand < cumsum( p ), 1 ); % discrete random sample

p = drand( p0 );
for i = 1:t
    dat(:,k+i) = chol(Q{p})'*randn(m,1) + A{p}*dat((i-1)*m + (1:m*k))';
    p = drand( T(:,p) );
end

dat = dat(:,k+1:end);
dat = dat + mx(:,ones(1,t));