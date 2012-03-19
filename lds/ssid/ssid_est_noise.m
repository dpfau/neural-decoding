function [Q R] = ssid_est_noise( e, C, A, s )
% Estimate the noise covariances for the linear dynamical system:
% x(t+1) = A*x(t) + B*u(t) + v(t)
% y(t)   = C*x(t) + D*u(t) + w(t)
% v(t) ~ N(0,Q)
% w(t) ~ N(0,R)
% By looking at the covariance between residuals at different time steps,
% where the residuals are the difference between the observed y(t) and the
% estimated y(t) assuming zero noise.
%
% Input:
%   e - the difference between the observed y and the y reconstructed from
%       a noiseless linear dynamical system
%   C - the estimated latent-state-to-output matrix
%   A - the estimated latent state evolution matrix
%   s - the maximum time lag for which we use the covariance.  If dim(x) <
%       dim(y) then asymptotically we'd never need s > 1, but in practice it
%       helps
%
% Output:
%   Q - Latent state noise, which introduces covariance between residuals
%       at different time steps
%   R - Output noise, which is uncorrelated from step to step

if nargin == 3
    s = 5;
end

CA = [];
ee = [];
for i = 1:s
    CA = [CA; C*A^i];
    ee = [ee; (e(:,1+i:end)*e(:,1:end-i)')/(size(e,2)-i)];
end
S = pinv(CA)*ee*pinv(C');
Q = S-A*S*A';
R = (e*e')/size(e,2) - C*S*C';