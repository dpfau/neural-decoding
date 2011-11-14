function [z P z1 P1] = kalman_filter(y, A, C, Q, R, P0)
% Forward pass of a Kalman smoother.  If used for smoothing, we also want
% to output P_{t+1|t}, which we denote P1 here.

z = zeros(size(P0,1),size(y,2));
P = zeros(size(P0,1),size(P0,2),size(y,2));
if nargout == 3
    z1 = zeros(size(z));
    P1 = zeros(size(P));
end

zt = zeros(size(P0,1),1);
Pt = P0;
for i = 1:size(y,2)
    % predict
    xt = A*xt;
    Pt = A*Pt*A' + Q;
    if nargout == 3
        z1(:,i) = zt;
        P1(:,:,i) = Pt;
    end
    
    % update
    Kt = Pt*C'*(C*PtC' + R)^-1; % Kalman gain
    xt = xt + Kt*(y(:,i) - C*xt);
    Pt = Pt - Kt*C*Pt;
    
    z(:,i) = zt;
    P(:,:,i) = Pt;
end