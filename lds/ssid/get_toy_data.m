addpath '/Users/davidpfau/Documents/Paninski Group/neural-decoding/util'
addpath '/Users/davidpfau/Documents/Paninski Group/neural-decoding/lds/plds'

m = 4; n = 5; l = 3;

u = randn(m,1000);

A = randn(n)/4;
B = randn(n,m);
C = randn(l,n)/25;

Q = diag(rand(n,1));

params = struct('A',A,'B',B,'C',C,'b',zeros(l,1),'Q',Q,'Q0',Q,'x0',zeros(n,1),'f',@(x,y) exp(x));
[x,y] = gen(params,1000);