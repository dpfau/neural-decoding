A = stabilize(randn(n));
B = randn(n,m);
C = randn(l,n);
D = randn(l,m);
x0 = randn(n,1);

K = randn(n+l); % output and process noise
K = K*K'; % make it positive definite

Q = K(1:n,1:n);
R = K(n+1:end,n+1:end);
S = K(1:n,n+1:end);

u = randn(m,1000);

[x,y] = gen(A,B,C,D,x0,u,Q,R,S);