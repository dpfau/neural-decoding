function [A Q T p0 mx] = switching_AR_EM( x, k, n, verbose, init )

if nargin < 4
    verbose = 0;
end
if nargin < 5
    init = 'rand';
end
eps = 1e-6;
A = cell(n,1);
Q = cell(n,1);
mx = mean( x, 2 );
x = x - repmat( mx, 1, size( x, 2 ) );

switch init
    case 'AR'
        [A0 Q0] = AR(x,x,k);
        for i = 1:n
            A{i} = A0;
            Q{i} = Q0;
        end
        T = (1-1e-2) * eye(n) + 1e-2/n; % set initial switching probability to 1/100, why not?
        p0 = ones(n,1)/n;
    case 'rand'
        q = 100;
        t = size( x, 2 );
        z = zeros( n, t-k );
        V = zeros( n );
        switches = [1, sort(ceil((t-k)*rand(1,q))), t-k];
        for i = 1:q+1
            z( ceil( n*rand ), switches(i):switches(i+1) ) = 1;
        end
        for i = 2:t-k
            V = V + z(:,i-1)*z(:,i)';
        end
        [A Q T p0] = switching_AR_M_step( x, z, V, k );
    otherwise
        error( 'Not a recognized initialization' );
end

ll_ = -Inf;
[z c l] = switching_AR_forward( x, A, Q, T, p0 );
ll = sum( log( c ) ); 
iter = 0;
while abs( ll - ll_ ) > eps || iter < 10
    ll_ = ll;
    [z V] = switching_AR_backward( x, A, T, z, c, l );
    [A Q T p0] = switching_AR_M_step( x, z, V, k );
    [z c l]  = switching_AR_forward( x, A, Q, T, p0 );
    ll = sum( log( c ) );
    iter = iter + 1;
    if verbose, fprintf( 'Iter: %i, LL = %d\n', iter, ll ); end
end