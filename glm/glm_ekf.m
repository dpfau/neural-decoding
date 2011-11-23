function [z V ll] = glm_ekf(y, A, C, f, grad, b, Q, z0, P0, varargin)
% Extended Kalman filter for online decoding.  Approximates p(z_t|y_1:t)
% where the generative model is
% x_t+1 = A*x_t + v_t, v_t ~ N(0,Q)
% y_t ~ Poiss( f(C*x_t + b) )

k = size(P0,1);
T = size(y,2);
for i = 1:2:length(varargin)
    eval([varargin{i} ' = varargin{' num2str(i+1) '};']);
end
if exist('u','var') && ~exist('B','var')
    error('Input-output parameters missing!')
end
if exist('B','var') && ~exist('u','var')
    error('Input data missing!');
end

z = zeros(k,T);
V = zeros(k,k,T);

ll = zeros(T,1); % log likelihood
zt = z0;
Pt = P0;

for i = 1:T
    R = f(C*zt + b);
    Rinv = diag(1./R);
    H = grad(C*zt + b)*ones(1,k) .* C;
    % Precision matrix of y(:,i) given y(:,1:i-1).
    if size(y,1) > 50
        % Same as (H*Pt*H' + R)^-1 by Woodbury lemma.
        T = Rinv*H;
        Sinv = Rinv - T*(Pt^-1 + H'*T)^-1*T';
    else
        Sinv = (H*Pt*H' + diag(R))^-1;
    end
    xt = y(:,i) - f(C*zt + b); % residual
    ll(i) = y(:,i)'*log(f(C*zt + b)) - sum(f(C*zt + b) + gammaln(y(:,i) + 1)); % log likelihood of one observation
    
    % update
    Kt = Pt*H'*Sinv; % Kalman gain
    zt = zt + Kt*xt;
    Vt = Pt - Kt*H*Pt;
    
    z(:,i) = zt;
    V(:,:,i) = Vt;
    
    % predict
    zt = A*zt;
    if exist('B','var')
        zt = zt + B*u(:,i);
    end
    Pt = A*Vt*A' + Q;
end