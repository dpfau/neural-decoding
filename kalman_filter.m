function [z P] = kalman_filter(y, A, C, Q, R, z0)

z = zeros(length(z0),size(y,2));
P = zeros(length(z0),length(z0),size(y,2));

zt = z0;
Pt = zeros(length(z0));
for i = 1:size(y,2)
    % predict
    xt = A*xt;
    Pt = A*Pt*A' + Q;
    
    % update
    Kt = Pt*C'*(C*PtC' + R); % Kalman gain
    xt = xt + Kt*(y(:,i) - C*xt);
    Pt = Pt - Kt*C*Pt;
    
    z(:,i) = zt;
    P(:,:,i) = Pt;
end