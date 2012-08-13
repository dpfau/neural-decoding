function [A Q T p0 mx] = switching_AR_EM( x, k, n )

eps = 1e-12;
A = cell(n,1);
Q = cell(n,1);
[A0 Q0 mx] = AR(x,x,k);
for i = 1:n
    A{i} = A0;
    Q{i} = Q0;
end
T = (1-1e-2) * eye(n) + 1e-2/n; % set initial switching probability to 1/100, why not?
p0 = ones(n,1)/n;
x = x - repmat( mx, 1, size( x, 2 ) );

ll_ = -Inf;
[z c l] = switching_AR_forward( x, A, Q, T, p0 );
ll = sum( log( c ) ); 
iter = 0;
while abs( ll - ll_ ) > eps
    ll_ = ll;
    [z V] = switching_AR_backward( x, A, T, z, c, l );
    [A Q T p0] = switching_AR_M_step( x, z, V, k );
    [z c l]  = switching_AR_forward( x, A, Q, T, p0 );
    ll = sum( log( c ) );
    iter = iter + 1;
    fprintf('Iter: %i, LL = %d\n',iter,ll);
end