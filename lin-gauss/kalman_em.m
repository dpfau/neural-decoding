function [A C Q R z0 V0 B D] = kalman_em(y,n,tol,varargin)
%% Initialize variables
for i = 1:2:length(varargin)
    eval([varargin{i} ' = varargin{' num2str(i+1) '};']);
end
m = size(y,1);
T = size(y,2);

vn = {'A','C','Q','R','z0','V0'};
nrow = [n, m, n, m, n, n];
ncol = [n, n, n, m, 1, n];
pd = [0,0,1,1,0,0];
for i = 1:6
    if ~exist(vn{i})
        eval([vn{i} ' = randn(nrow(i),ncol(i));']);
        if pd(i), eval([vn{i} ' = ' vn{i} '*' vn{i} ''';']); end
    else
        eval(['assert( size(' vn{i} ',1) == nrow(i), ''' vn{i} ' has incorrect number of rows.'');']);
        eval(['assert( size(' vn{i} ',2) == ncol(i), ''' vn{i} ' has incorrect number of columns.'');']);
    end
end

if exist('u','var')
    if ~exist('B','var')
        B = randn(n,size(u,1)); 
    else
        assert( size(B,1) == n, 'B has incorrect number of rows' );
        assert( size(B,2) == size(u,1), 'B has incorrect number of columns' );
    end
    if exist('D','var')
        assert( size(D,1) == size(y,1), 'D has incorrect number of rows' );
        assert( size(D,2) == size(u,1), 'D has incorrect number of columns' );
    elseif nargout == 8
        D = randn(m,size(u,1));
    end
elseif nargout > 6
    error('Missing input data!');
end

%% EM loop
ll0 = -Inf;
[z V lls VV] = kalman_smoother(y,A,C,Q,R,z0,V0,varargin);
ll = sum(lls);
while abs(ll - ll0) > tol
    z0 = z(:,1);
    V0 = V(:,:,1);

    Ptt1  = sum(VV(:,:,2:end),3) + z(:,2:end)*z(:,1:end-1)';
    Pt1t1 = sum(V(:,:,1:end-1),3) + z(:,1:end-1)*z(:,1:end-1)';
    Ptt   = sum(V(:,:,2:end),3) + z(:,2:end)*z(:,2:end)';
    if ~exist('u','var')
        A = Ptt1*Pt1t1^-1;
        Q = 1/(T-1)*(Ptt - A*Ptt1');
    else
        Pinv = Pt1t1^-1;
        uz = u(:,1:end-1)*z(:,1:end-1)';
        zu = z(:,2:end)*u(:,1:end-1)';
        B = (zu - Ptt1*Pinv*uz')*(u(:,1:end-1)*u(:,1:end-1)' - uz*Pinv*uz')^-1;
        A = (Ptt1 - B*uz)*Pinv;
        Q = 1/(T-1)*(Ptt - A*Ptt1' - B*zu');
    end
    
    if ~exist('D','var')
        C = (y*z')*(sum(V,3) + z*z')^-1;
        R = 1/T*(y*y' - C*z*y');
    else
        Pinv = (sum(V,3) + z*z')^-1;
        D = (y*u' - y*z'*Pinv*z*u')*(u*u' - u*z'*Pinv*z*u')^-1;
        C = (y*z' - D*u*z')*Pinv;
        R = 1/T*(y*y' - C*z*y' - D*u*y');
    end
    
    fprintf('Data log likelihood: %d\n',ll);
    ll0 = ll;
    [z V lls VV] = kalman_smoother(y,A,C,Q,R,z0,V0);
    ll = sum(lls);
end