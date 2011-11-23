function [x y] = gen_glm(A,B,C,f,b,x0,u,Q)

n = size(x0,1);
x = zeros(n,size(u,2));

if isempty(Q)
    Q = zeros(n);
end

x(:,1) = x0;
E = chol(Q);
for i = 1:size(u,2)
    e = E*randn(n,1);
    if i < size(u,2)
        x(:,i+1) = A*x(:,i) + B*u(:,i) + e(1:n);
    end
end

y = poissrnd( f( C*x + b ) );