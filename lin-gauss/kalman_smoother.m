function [z V ll VV] = kalman_smoother(y, A, C, Q, R, z0, P0)
% Really the Rauch-Tung-Striebel smoother, but that's too jargony for me
% The JV term is needed for EM, but may be omitted

[z_ V_ ll VV_ P] = kalman_filter(y, A, C, Q, R, z0, P0); % forward pass

z = zeros(size(z_));
V = zeros(size(V_));
if nargout > 3
    VV = zeros(size(VV_));
end

zt = z_(:,end);
Vt = V_(:,:,end);

for i = size(y,2)-1:-1:1
    z(:,i+1) = zt;
    V(:,:,i+1) = Vt;
    
    Lt = V_(:,:,i)*A'*P(:,:,i)^-1;
    zt = z_(:,i) + Lt*(zt - A*z_(:,i));
    if nargout > 3
        VV(:,:,i+1) = VV_(:,:,i+1) + (Vt - V_(:,:,i+1))*(V_(:,:,i+1))^-1*VV_(:,:,i+1);
    end
    Vt = V_(:,:,i) + Lt*(Vt - P(:,:,i))*Lt';
end
z(:,1) = zt;
V(:,:,1) = Vt;