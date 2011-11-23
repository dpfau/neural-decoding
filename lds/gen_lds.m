function [x y] = gen_lds(A,B,C,D,x0,u,Q,R,S)

x = zeros(size(x0,1),size(u,2));
y = zeros(size(D,1),size(u,2));

m = size(u,1);
l = size(D,1);
n = size(x0,1);

if isempty(Q)
    Q = zeros(n);
end
if isempty(R)
    R = zeros(l);
end
if isempty(S)
    S = zeros(n,l);
end

if nnz([Q, S; S', R]) == 0
    E = zeros(size([Q, S; S', R]));
else
    E = chol([Q, S; S', R]); % sqrt of noise covariance
end
x(:,1) = x0;
for i = 1:size(u,2)
    e = E*randn(n+l,1);
    y(:,i) = C*x(:,i) + D*u(:,i) + e(n+1:end);
    if i < size(u,2)
        x(:,i+1) = A*x(:,i) + B*u(:,i) + e(1:n);
    end
end