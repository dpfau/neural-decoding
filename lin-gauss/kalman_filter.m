function [z V ll VV P] = kalman_filter(y, A, C, Q, R, z0, P0, varargin)
% Forward pass of a Kalman smoother.  If used for smoothing, we also want
% to output P_{t+1|t}, which we denote P here.
% Also includes optional u, B and D for the input-output case

for i = 1:2:length(varargin)
    eval([varargin{i} ' = varargin{' num2str(i+1) '};']);
end
if exist('u','var') && ~exist('B','var') && ~exist('D','var')
    error('Input-output parameters missing!')
end
if (exist('B','var') || exist('D','var')) && ~exist('u','var')
    error('Input data missing!');
end

z = zeros(size(P0,1),size(y,2));
V = zeros(size(P0,1),size(P0,2),size(y,2));
if nargout > 3
    VV = zeros(size(V));
end
if nargout > 4
    P = zeros(size(V));
end

ll = zeros(size(y,2),1); % log likelihood
zt = z0;
Pt = P0;

Rinv = R^-1; % store for fast matrix inversion
for i = 1:size(y,2)
    % Precision matrix of y(:,i) given y(:,1:i-1).
    if size(y,1) > 50
        % Same as (C*Pt*C' + R)^-1 by Woodbury lemma.
        T = Rinv*C;
        Sinv = Rinv - T*(Pt^-1 + C'*T)^-1*T';
    else
        Sinv = (C*Pt*C' + R)^-1;
    end
    xt = y(:,i) - C*zt; % residual
    if exist('D','var')
        xt = xt - D*u(:,i);
    end
    ll(i) = - 0.5*( xt'*Sinv*xt + size(y,1)*log( 2*pi ) - log( det( Sinv ) ) ); % log likelihood of one observation
    
    % update
    Kt = Pt*C'*Sinv; % Kalman gain
    zt = zt + Kt*xt;
    if nargout > 3
        if i > 1
            VV(:,:,i) = A*Vt - Kt*C*A*Vt;
        else
            VV(:,:,i) = A*P0 - Kt*C*A*P0;
        end
    end
    Vt = Pt - Kt*C*Pt;
    
    z(:,i) = zt;
    V(:,:,i) = Vt;
    
    % predict
    zt = A*zt;
    if exist('B','var')
        zt = zt + B*u(:,i);
    end
    Pt = A*Vt*A' + Q;
    if nargout > 4
        P(:,:,i) = Pt;
    end
end