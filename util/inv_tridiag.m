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
v = zeros(1,N);
u = zeros(1,N);

lv = zeros(1,N);
lu = zeros(1,N);

d = zeros(1,N); % Diagonal of the UL decomposition
d(end) = x.diag(end);
for i = N-1:-1:1
    d(i) = x.diag(i) - x.off_diag(i)^2/d(i+1);
end

c = zeros(1,N); % Diagonal of the LU decomposition
c(1) = x.diag(1);
for i = 2:N
    c(i) = x.diag(i) - x.off_diag(i-1)^2/c(i-1);
end

lv(1) = -log(d(1));
for i = 2:N
    lv(i) = lv(i-1) + log(x.off_diag(i-1)) - log(d(i));
end

lu(end) = -log(c(end)) - lv(end);
for i = 1:N-1
    lu(end-i) = lu(end-i+1) + log(x.off_diag(end-i+1)) - log(c(end-i));
end

y = struct('diag',real(exp(lu+lv)),'off_diag',-real(exp(lu(1:end-1)+lv(2:end))));