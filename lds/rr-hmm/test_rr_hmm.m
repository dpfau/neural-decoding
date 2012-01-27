n = 200; % number of observables
k = 5; % rank of RR-HMM
D = 3; % dimension of data space
c = 10*randn(D,n); % kernel centers
m = 30; % number of latent states, higher than the rank of T

T = rand(m); % transition matrix
[u,s,v] = svd(T);
T = u(:,1:k)*s(1:k,1:k)*v(:,1:k)'; % reduce rank
T = T./(ones(m,1)*sum(T)); % normalize
O = rand(n,m); % observation matrix
O = O./(ones(m,1)*sum(O)); % normalize
p = rand(m,1);
p = p/sum(p); % initial probability

Q = randn(D);
Q = chol(Q*Q');

A_x = zeros(m,m,n);
for i = 1:n
    A_x(:,:,i) = T*diag(O(i,:));
end

data = zeros(D,10000);
b_t = p;
for i = 1:size(data,2)
    data(:,i) = generate( b_t, Q, c, ones(m,1), A_x );
    b_t = update( b_t, data(:,i), @(x) exp( -column_squared_norm(x) ), Q^-1, c, 1, ones(m,1), A_x );
end

[b_1 b_inf B_x] = est_RR_HMM( 