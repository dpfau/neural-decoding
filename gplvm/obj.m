function f = obj(params,y,d)

D = size(y,1);
N = size(y,2);
a = params(1);
b = params(2);
c = params(3);
w = params(4:D+3);
z = reshape(params(D+4:end),d,N);

K = kernel(z,a,b,c);
wy = diag(w)*y;
Ki = K^-1; % This is the biggest impediment to scaling
f = D*sum(log(diag(chol(K)))) ...
    + 1/2*sum(diag(wy*Ki*wy')) ...
    + 1/2*z(:)'*z(:) ...
    + log(a) + log(b) + log(c) ...
    - N*sum(log(w));