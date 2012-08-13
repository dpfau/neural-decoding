function [A Q mx ll] = AR(x,y,k)

[m n] = size(x);
q = size(y,2);
mx = mean(x,2);
x = x - repmat(mx,1,n);
y = y - repmat(mx,1,q);
x_ = zeros(k*m,n-k);
y_ = zeros(k*m,q-k);
for i = 1:k
    x_((1:m) + m*(k-1),:) = x(:,i:end-k+i-1);
    y_((1:m) + m*(k-1),:) = y(:,i:end-k+i-1);
end
A = x(:,k+1:end)*pinv(x_);
r = x(:,k+1:end) - A*x_;
Q = r*r'/(n-k);
r = y(:,k+1:end) - A*y_;
ll = (q-k)*(-m/2*log(2*pi) - sum(log(diag(chol(Q)))));
for i = 1:q-k
    ll = ll - 1/2*r(:,i)'*(Q\r(:,i));
end
ll = ll/(q-k);