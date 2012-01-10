function [A C Q R my mz] = fit_kalman(y, z, k)
% Fits a linear-Gaussian model from fully observed data:
% z(t+1) = A*z(t-k+1:t) + v(t)
% y(t)   = C*z(t-k+1:t) + w(t)
% v(t) ~ N(0,Q)
% w(t) ~ N(0,R)
%
% my and mz are mean terms subtracted from y and z, for data with nonzero
% mean.
% David Pfau, 2011

if ~iscell(y)
    y = {y};
end

if ~iscell(z)
    z = {z};
end

assert( length(y) == length(z), 'Must be same number of input and output sequences' );
n = 0;
m = size(z{1},1);
l = size(y{1},1);
for t = 1:length(y)
    assert( size(y{t},2) == size(z{t},2), 'Input and output data must be same length' );
    assert( size(y{t},1) == l, 'Wrong number of output dimensions' );
    assert( size(z{t},1) == m, 'Wrong number of input dimensions' );
    n = n + size(y{t},2) - k + 1;
end

% stack time-shifted versions of z to fit AR(k) model
z_ = [];
y_ = [];
for t = 1:length(z)
    z__ = zeros(m*k,size(z{t},2)-k+1);
    for i = 1:k
        z__((i-1)*m + (1:m),:) = z{t}(:,i:end-k+i);
    end
    z_ = [z_, z__];
    y_ = [y_, y{t}(:,k:end)];
end

% subtract means from data
mz = mean(z_,2);
z_ = z_ - mz*ones(1,n);
my = mean(y_,2);
y_ = y_ - my*ones(1,n);

A = z_(:,2:end) * pinv( z_(:,1:end-1) );
C = y_ * pinv( z_ );

Ares = z_(:,2:end) - A*z_(:,1:end-1);
Q = Ares*Ares'/n;

Cres = y_ - C*z_;
R = Cres*Cres'/n;