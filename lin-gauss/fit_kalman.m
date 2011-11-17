function [A C Q R] = fit_kalman(y, z)
% Fits a linear-Gaussian model from fully observed data:
% z_t+1 = Az_t + v_t
% y_t   = Cz_t + w_t
% v_t ~ N(0,Q)
% w_t ~ N(0,R)
% David Pfau, 2011

assert( size(y,2) == size(z,2), 'Input and output data must be same length' );
n = size(y,2);

A = z(:,2:end) * pinv( z(:,1:end-1) );
C = y * pinv( z );

Ares = z(:,2:end) - A*z(:,1:end-1);
Q = Ares*Ares'/n;

Cres = y - C*z;
R = Cres*Cres'/n;