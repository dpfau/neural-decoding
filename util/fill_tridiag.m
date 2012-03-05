function y = fill_tridiag( x )
% For a given struct containing the relevant fields of a tridiagonal or
% block tridiagonal matrix, return the full matrix constructed from those
% fields

if isfield(x,'diag')
    N = length(x.diag);
    y = diag(x.diag);
    y(2:N+1:end) = x.off_diag;
    y(N+1:N+1:end) = x.off_diag;
elseif isfield(x,'diag_upper')
    N = size(x.diag_center,2); % number of block-rows and block-columns
    k = size(x.diag_center,1); % size of each square block
    y = zeros(N*k);
    for i = 1:N-1
        y((i-1)*k + (1:k), i*k + (1:k)) = x.off_diag;
        y(i*k + (1:k), (i-1)*k + (1:k)) = x.off_diag';
        y((i-1)*k + (1:k), (i-1)*k + (1:k)) = y((i-1)*k + (1:k), (i-1)*k + (1:k)) + x.diag_upper;
        y(i*k + (1:k), i*k + (1:k)) = y(i*k + (1:k), i*k + (1:k)) + x.diag_lower;
    end
    for i = 1:N
        y((i-1)*k + (1:k), (i-1)*k + (1:k)) = y((i-1)*k + (1:k), (i-1)*k + (1:k)) + x.diag_left*diag(x.diag_center(:,i))*x.diag_right;
    end
else
    error('Not a recognized format for block tridiagonal matrices')
end