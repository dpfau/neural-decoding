function dat = gen_AR( A, Q, mx, t )

m = size(A,1);
k = floor(size(A,2)/m);
dat = zeros( m, t + k );
e = chol(Q)'*randn( size(A,1), t + k );
dat(:,1:k) = e(:,1:k); % initialize with noise

for i = 1:t
    dat(:,k+i) = e(:,k+i) + A*dat((i-1)*m + (1:m*k))';
end

dat = dat(:,k+1:end);
dat = dat + mx(:,ones(1,t));