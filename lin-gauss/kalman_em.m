function [A C Q R z0 V0] = kalman_em(y,n,tol,varargin)

ll0 = -Inf;

for i = 1:2:length(varargin)
    eval([varargin{i} ' = varargin{' num2str(i+1) '};']);
end

% Note that random initializations are often very bad for EM
if ~exist('A','var'), A = randn(n); end
if ~exist('C','var'), C = randn(size(y,1),n); end
if ~exist('Q','var')
    Q = randn(n);
    Q = Q*Q'; % make positive definite
end
if ~exist('R','var')
    R = randn(size(y,1));
    R = R*R';
end
if ~exist('z0','var'), z0 = randn(n,1); end
if ~exist('V0','var'), V0 = Q; end

[z V lls VV] = kalman_smoother(y,A,C,Q,R,z0,V0);
ll = sum(lls);
while abs(ll - ll0) > tol
    z0 = z(:,1);
    V0 = V(:,:,1);

    Ptt1 = sum(VV(:,:,2:end),3) + z(:,2:end)*z(:,1:end-1)';
    A = Ptt1*(sum(V(:,:,1:end-1),3) + z(:,1:end-1)*z(:,1:end-1)')^-1;
    Q = 1/(size(y,2)-1)*(sum(V(:,:,2:end),3) + z(:,2:end)*z(:,2:end)' - A*Ptt1');
    
    C = (y*z')*(sum(V,3) + z*z')^-1;
    R = 1/size(y,2)*(y*y' - C*z*y');
    
    fprintf('Data log likelihood: %d\n',ll);
    ll0 = ll;
    [z V lls VV] = kalman_smoother(y,A,C,Q,R,z0,V0);
    ll = sum(lls);
end