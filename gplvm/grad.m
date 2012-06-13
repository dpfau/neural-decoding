function g = grad(params,y,d)

D = size(y,1);
N = size(y,2);
a = params(1);
b = params(2);
c = params(3);
w = params(4:D+3);
z = reshape(params(D+4:end),d,N);

[K g e] = kernel(z,a,b,c);
wy = diag(w)*y;
Ki = K^-1; % This is the biggest impediment to scaling

dK = (-(Ki*wy')*(wy*Ki) + D*Ki)/2;
dz1 = -c*(tprod(K,[2 3],z,[1 2])-tprod(K,[2 3],z,[1 3]));
dz = 2*tprod(dK,[2 -1],dz1,[1 2 -1]) + z;
dw = w.*diag(y*Ki*y')-N./w;
g = [ trace(dK*e) + 1/a; ...
    trace(-dK/b^2) + 1/b; ...
    trace(-1/2*dK*(g.*K)) + 1/c; ...
    dw(:); ...
    dz(:) ];