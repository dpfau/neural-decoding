function y = inv_block_tridiag( x )
% Following A. Asif and J. M. F. Moura, "Block Matrices with
% L-Block Banded Inverse: Inversion Algorithms", we calculate the
% block-diagonal and block-off-diagonal of the inverse Hessian for the
% Poisson-LDS path posterior probability, which gives us the covariance
% sufficient statistics needed to perform the M-step in learning.
%
% David Pfau, 2012

N = size(x.diag_center,2); % Number of blocks
k = size(x.diag_center,1); % size of each block
UtU = zeros(k,k,N); % U'*U, where U is a diagonal block of the Cholesky decomposition of the block-tridiagonal matrix
UiU1 = zeros(k,k,N-1); % U^-1*U1, where U is a diagonal block and U1 an off-diagonal block of the block-tridiagonal matrix
U1tU1 = zeros(k,k,N-1); % U1'*U1, U1 is off-diagonal block

A = x.diag_upper + x.diag_left*diag(x.diag_center(:,1))*x.diag_right;
UtU(:,:,1) = A;
for i = 1:N-1
    UiU1(:,:,i) = UtU(:,:,i)^-1*x.off_diag;
    U1tU1(:,:,i) = UiU1(:,:,i)'*UtU(:,:,i)*UiU1(:,:,i);
    UtU(:,:,i+1) = x.diag_lower + x.diag_left*diag(x.diag_center(:,i+1))*x.diag_right - U1tU1(:,:,i);
    if i < N-1
        UtU(:,:,i+1) = UtU(:,:,i+1) + x.diag_upper;
    end
end

y = struct('diag',zeros(k,k,N),'off_diag',zeros(k,k,N-1));
y.diag(:,:,end) = UtU(:,:,end)^-1;
for i = N-1:-1:1
    y.off_diag(:,:,i) = UiU1(:,:,i)*y.diag(:,:,i+1);
    y.diag(:,:,i) = UtU(:,:,i)^-1 - y.off_diag(:,:,i)*UiU1(:,:,i)';
end