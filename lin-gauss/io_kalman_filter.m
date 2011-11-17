function [z P z1 P1] = io_kalman_filter(y, u, A, B, C, D, Q, R, P0)
% Forward pass of a Kalman smoother with input and output

z = zeros(size(P0,1),size(y,2));
P = zeros(size(P0,1),size(P0,2),size(y,2));
if nargout == 4
    z1 = zeros(size(z));
    P1 = zeros(size(P));
end

zt = zeros(size(P0,1),1);
Pt = P0;
for i = 1:size(y,2)
    % predict
    zt = A*zt + B*u(:,i);
    Pt = A*Pt*A' + Q;
    if nargout == 4
        z1(:,i) = zt;
        P1(:,:,i) = Pt;
    end
    
    % update
    Kt = Pt*C'*(C*Pt*C' + R)^-1; % Kalman gain
    zt = zt + Kt*(y(:,i) - C*zt - D*u(:,i));
    Pt = Pt - Kt*C*Pt;
    
    z(:,i) = zt;
    P(:,:,i) = Pt;
end