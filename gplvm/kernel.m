function [K d e] = kernel(z,a,b,c)

N = size(z,2);
zz = z'*z;
d = diag(zz)*ones(1,N) - 2*zz + ones(N,1)*diag(zz)';
e = exp( -c/2*d );
K = a*e + eye(N)/b;