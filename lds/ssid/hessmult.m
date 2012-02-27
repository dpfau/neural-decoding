function hv = hessmult(Hinfo,v)
% Multiplication by the Hessian of the objective for the augmented
% negative log likelihood step in ADMM

N = size(Hinfo.eyb,2);
hv = zeros(size(v));
for i = 1:size(v,2)
    v2 = reshape(v(:,i),size(Hinfo.eyb,1),N+1);
    hv2 = [ (Hinfo.lam*Hinfo.eyb + Hinfo.rho).*v2(:,1:N) + Hinfo.lam*Hinfo.eyb.*(v2(:,end)*ones(1,N)), ...
            Hinfo.lam*sum(Hinfo.eyb.*v2(:,1:N),2) + (Hinfo.lam*sum(Hinfo.eyb,2) + Hinfo.rho).*v2(:,end) ];
    hv(:,i) = hv2(:);
end