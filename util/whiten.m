function [y m c] = whiten(x)

m = mean(x);
y = x(:,m~=0);
m = m(m~=0);
y = y - m(ones(1,size(y,1)),:);
c = (y'*y)/size(y,1);
y = (chol(c)'\y')';