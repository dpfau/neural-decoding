function [A C Q R z0 V0] = kalman_em(y, n, tol)

ll0 = -Inf;

A = randn(n);
C = randn(size(y,1),n);
Q = randn(n);
Q = Q*Q'; % make positive definite
R = randn(size(y,1));
R = R*R';
z0 = randn(n,1);
V0 = Q;

[z V ll JV] = kalman_smoother(y,A,C,Q,R,z0,V0);
while abs(ll - ll0) > tol
    z0 = z(:,1);
    V0 = V(:,:,1);
    
    Ezt1zt = JV' + z(2:end)*z(1:end-1)';
    Eztzt = sum(V(:,:,1:end-1),3) + z(1:end-1)*z(1:end-1)';
    A = Ezt1zt*Eztzt^-1;
    Q = 1/(size(y,2)-1)*(sum(V(:,:,2:end),3) + z(2:end)*z(2:end)' - A*Ezt1zt' - Ezt1zt*A' + A*Eztzt*A');
    
    C = y*z'*(sum(V,3) + z*z')^-1;
    R = 1/size(y,2)*(y*y' - C*z*y' - y*z'*C' + C*(sum(V,3) + z*z')*C');
    
    fprintf('Data log likelihood: %d\n',ll);
    ll0 = ll;
    [z V ll JV] = kalman_smoother(y,A,C,Q,R,z0,V0);
end