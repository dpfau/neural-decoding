function [b_1 b_inf B_x] = est_RR_HMM( dat, r, k, c1, c2, c3, l)
% Estimate a reduced-rank hidden markov model, a la Siddiqi, Boots and
% Gordon.
%
% Input:
%   dat - cell array of 2D data, each column is one data point
%   r - rank of the model to be estimated
%   k - kernel function
%   c1 - kernel centers for the past sequence
%   c2 - kernel centers for the present point
%   c3 - kernel centers for the future sequence
%   l - kernel bandwidth

assert( mod( size( c1, 1 ), size( dat{1}, 1 ) ) == 0, 'Past kernel centers are wrong size' );
assert( mod( size( c3, 1 ), size( dat{1}, 1 ) ) == 0, 'Future kernel centers are wrong size' );
assert( size( c2, 1 ) == size( dat{1}, 1 ), 'Present kernel centers are wrong size' );

n = size(c1,1)/size(dat{1},1); % number of past steps used for prediction
m = size(c3,1)/size(dat{1},1); % number of future steps used for prediction

k1 = size(c1,2);
k2 = size(c2,2);
k3 = size(c3,2);

phi  = [];
psi  = [];
xi   = [];
zeta = [];
for i = 1:length(dat)
    dat_past = block_hankel(dat{i},1,n,size(dat{i},2)-m-1);
    dat_future = block_hankel(dat{i},n+1,m,size(dat{i},2));
    N = size(dat{i},2) - n - m;
    assert( size(dat_past,2) == N );
    assert( size(dat_future,2) == N );
    
    phi1 = zeros(k1,N);
    for j = 1:k1
        phi1(j,:) = k( dat_past - c1(:,j)*ones(1,N) );
    end
    phi = [phi phi1];
    
    psi1  = zeros(k2,N);
    zeta1 = zeros(k2,N);
    for j = 1:k2
        psi1(j,:)  = k( dat{i}(:,n+1:end-m) - c2(:,j)*ones(1,N) );
        zeta1(j,:) = k( (dat{i}(:,n+1:end-m) - c2(:,j)*ones(1,N))/l );
    end
    psi  = [psi psi1];
    zeta = [zeta zeta1];
    
    xi1 = zeros(k3,N);
    for j = 1:k3
        xi1(j,:) = k( dat_future - c3(:,j)*ones(1,N) );
    end
    xi = [xi xi1];
end
phi  = phi./(ones(k1,1)*sum(phi));
psi  = psi./(ones(k2,1)*sum(psi));
xi   = xi./(ones(k3,1)*sum(xi));
zeta = zeta./(ones(k2,1)*sum(zeta)); % Normalize all the feature vectors

P1 = mean(phi,2);

P21 = zeros(k2,k1);
for i = 1:N
    P21 = P21 + psi(:,i)*phi(:,i)';
end
P21 = P21/N;

P3x1 = zeros(k3,k1,k2);
for i = 1:k2
    for j = 1:N
        P3x1(:,:,i) = P3x1(:,:,i) + zeta(i,j)*xi(:,j)*phi(:,j)';
    end
end
P3x1 = P3x1/N;

U = svd(P21);
b_1 = U(:,1:r)'*P1;
b_inf = pinv( P21'*U(:,1:r) )*P1;
B_x = zeros(r,r,k2);
UP = pinv( U(:,1:r)'*P21 );
for i = 1:k2
    B_x(:,:,i) = U(:,1:r)'*P3x1(:,:,i)*UP;
end