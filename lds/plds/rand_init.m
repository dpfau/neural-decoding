function [params map data] = rand_init( k, N, T, t )
% For now, a random initialization of a Poisson-LDS model.  Definitely
% should implement more justified initialization later.
% k - number of latent dimensions
% N - number of output dimensions
% T - number of time steps of generated data
% t - length of history term

if nargin < 4
    t = 0;
end

A = randn(k);
[u,s,v] = svd(A);
s(s>0.99) = 0.99;
A = u*s*v';

C = randn(N,k);
Q = 0.2*randn(k);
Q = Q*Q';
b = 0.1*randn(N,1);
f = @(x,y) exp(x);
D = randn(N,N*t);

params = struct('A',A,'C',C,'Q',Q,'b',b,'f',f,'x0',zeros(k,1),'Q0',Q,'D',D,'t',t);
[data,map] = gen( params, T );