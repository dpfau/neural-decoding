function y = inv_tridiag( x )
% Given the diagonal and off-diagonal of a symmetric tridiagonal matrix,
% returns the diagonal and off-diagonal term of the inverse.  In the case
% of interest to us, where the tridiagonal is the precision matrix (that is
% the Hessian of the log likelihood), the diagonal and off-diagonal of the 
% inverse are the marginal variance of individual time points and
% covariance of adjacent time points, which are sufficient statistics for
% the M-step of EM for the latent phase model.  Both input and output are
% structs with fields 'diag' and 'off_diag', with obvious meaning.
%
% David Pfau

N = length(x.diag);
v = zeros(N,1);
u = zeros(N,1);

d = zeros(N,1); % Diagonal of the UL decomposition
d(end) = x.diag(end);
for i = N-1:-1:1
    d(i) = x.diag(i) - x.off_diag(i)^2/d(i+1);
end

c = zeros(N,1); % Diagonal of the LU decomposition
c(1) = x.diag(1);
for i = 2:N
    c(i) = x.diag(i) - x.off_diag(i-1)^2/c(i-1);
end

v(1) = 1/d(1);
k = 1;
for i = 2:N
    k = -k;
    v(i) = k*prod(x.off_diag(1:i-1))/prod(d(1:i));
end

u(end) = 1/c(end)/v(end);
k = 1;
for i = 1:N-1
    k = -k;
    u(end-i) = k*prod(x.off_diag(end-i+1:end))/prod(c(end-i:end))/v(end);
end

y = struct('diag',u.*v,'off_diag',u(1:end-1).*v(2:end));