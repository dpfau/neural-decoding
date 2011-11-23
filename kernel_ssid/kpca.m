function [a d] = kpca(x,k)
% Kernel pca

n = size(x,2);
K = zeros(n);
% construct kernel
for i = 1:n
    for j = 1:n
        K(i,j) = k(x(:,i),x(:,j));
    end
end

% center kernel
K = K - 1/n*ones(n)*K - 1/n*K*ones(n) + 1/n^2*ones(n)*K*ones(n);
[a,d] = eig(K);
for i = 1:n
    a(:,i) = a(:,i)/(a(:,i)'*K*a(:,i));
end