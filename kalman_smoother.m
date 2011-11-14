function [z P] = kalman_smoother(y, A, C, Q, R, P0)

[z_ P_ z1 P1] = kalman_filter(y, A, C, Q, R, P0); % forward pass

z = zeros(size(z_));
P = zeros(size(P_));

zt = z_(:,end);
Pt = P_(:,end);

for i = size(y,2):-1:2
    z(:,i) = zt;
    P(:,:,i) = Pt;
    
    Lt = P_(:,:,i-1)*A'*P1(:,:,i)^-1;
    zt = z_(:,i-1) + Lt*(zt - z1(:,i));
    Pt = P_(:,:,i-1) + Lt*(Pt - P1(:,:,i))*Lt';
end
z(:,1) = zt;
P(:,:,1) = Pt;