A = randn(n,m);
B = randn(n,m);

[u,s,v] = svd(A);
[uu,ss,vv] = svd(B);

trace(A*vv*ss'*uu') - trace(A*v*ss'*u')