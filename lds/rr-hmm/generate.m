function x_t1 = generate( b_t, Q, c, b_inf, B_x )
% Given a current belief state, generate an observation from the predictive
% distribution
%
% Inputs:
%   b_t - current belief state
%   Q - square-root covariance of the kernels.  Includes kernel bandwidth
%       implicitly
%   c - kernel centers
%   b_inf, B_x - parameters of the observable HMM representation 
%
% Output:
%   x_t1 - generated data point

assert( size(c,2) == size(B_x,3) );

prob = zeros(size(c,2),1);
for i = 1:size(c,2)
    prob(i) = b_inf'*B_x(:,:,i)*b_t;
end
idx = find( cumsum(prob) > rand*sum(prob), 1 );

x_t1 = c(:,idx) + Q*randn(size(Q,2),1);