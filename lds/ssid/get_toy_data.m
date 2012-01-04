addpath '/Users/davidpfau/Documents/Paninski Group/git-repo/glm'

m = 4; n = 5; l = 3;

u = randn(m,1000);

A = randn(n)/4;
B = randn(n,m);
C = randn(l,n)/25;

Q = diag(rand(n,1));

[x,y] = gen_glm(A,B,C,@exp,zeros(l,1),zeros(n,1),u,Q);