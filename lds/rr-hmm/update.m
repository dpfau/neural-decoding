function [b_t1 like] = update( b_t, x, k, c, l, b_inf, B_x )
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
%   like - likelihood of x given b_t
%
% David Pfau, 2012

sig = zeros(size(c,2),1);
for i = 1:length(sig)
    sig(i) = k( (x - c(:,i))/l );
end
sig = sig/sum(sig);

B_sig = zeros(size(B_x,1),size(B_x,2));
likenorm = 0; % denominator for the likelihood term
for i = 1:length(sig)
    B_sig = B_sig + sig(i)*B_x(:,:,i);
    likenorm = likenorm + b_inf'*B_x(:,:,i)*b_t;
end

norm = b_inf'*B_sig_b_t; % normalizer for the update, numerator for the likelihood term
b_t1 = B_sig*b_t / norm;
like = norm / likenorm;