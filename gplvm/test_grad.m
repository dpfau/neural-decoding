z = randn(5,20);
a = 1;
b = 2;
c = 3;

K = kernel(z,a,b,c);
f = log(det(K));