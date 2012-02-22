function [map params] = rand_init( k, N, T )
% For now, a random initialization of a Poisson-LDS model.  Definitely
% should implement more justified initialization later

A = randn(k);
[u,s,v] = svd(A);
s(s>0.99) = 0.99;
A = u*s*v';

C = randn(N,k);
Q = 0.2*randn(k);
Q = Q*Q';
b = 0.1*randn(N,1);
f = @(x,y) exp(x);

params = struct('A',A,'C',C,'Q',Q,'b',b,'f',f);
[~,map] = gen( params, T, zeros(k,1) );