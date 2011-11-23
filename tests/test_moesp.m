function test_moesp( moesp, m, n, l, i, N, e )
% test different implementations of subspace id
% moesp( y, u, i, N ) - a moesp implementation that returns estimates of
% A,B,C,D and x0
% m - # of input dimensions
% n - # of hidden dimensions
% l - # of output dimensions
% i - # of block-Hankel rows (should be greater than n to work properly)
% N - # of time steps used for reconstruction (should be greater than i)

A = stabilize(randn(n));
B = randn(n,m);
C = randn(l,n);
D = randn(l,m);
x0 = randn(n,1);

K = e*randn(n+l); % output and process noise
K = K*K'; % make it positive definite

Q = K(1:n,1:n);
R = K(n+1:end,n+1:end);
S = K(1:n,n+1:end);

u = randn(m,2000);

[x,y] = gen(A,B,C,D,x0,u,Q,R,S);

test_dat = [u', y'];

save test_dat.mat test_dat

[At,Bt,Ct,Dt,x0t] = moesp( y, u, i, N );

[xt,yt] = gen(At,Bt,Ct,Dt,x0t,u,[],[],[]);

for t = 1:l
    subplot(l,1,t);
    plot(1:2000,y(t,:),1:2000,yt(t,:));
end

