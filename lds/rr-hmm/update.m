function b_t1 = update( b_t, x, k, c, l, b_inf, B_x )
% Update the belief state one step for an RR-HMM
%
% Inputs:
%   b_t - belief state at time t
%   x   - observation at time t
%   k   - kernel function
%   c   - 2D array of kernel centers
%   l   - kernel bandwidth
%   b_inf - parameter of observable HMM representation
%   B_x - parameter of observable HMM representation
%
% Output:
%   b_t1 - belief state at time t+1
%
% David Pfau, 2012

sig = zeros(size(c,2),1);
for i = 1:length(sig)
    sig(i) = k( (x - c(:,i))/l );
end
sig = sig/sum(sig);

B_sig = zeros(size(B_x,1),size(B_x,2));
for i = 1:length(sig)
    B_sig = B_sig + sig(i)*B_x(:,:,i);
end

b_t1 = B_sig*b_t / b_inf'*B_sig*b_t;