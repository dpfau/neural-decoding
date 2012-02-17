function [map prec ll] = e_step( data, params, init )
% Given data and parameters for a Poisson-LDS model, find the Laplace
% approximation to the expected log likelihood by first finding the MAP
% path, then taking a quadratic approximation of the log likelihood around
% that path.
%
% Input:
% data - vector of observed data
% params - struct of parameters in latent-phase model, with fields:
% init - initial value for latent variables
% 
% Output:
% map - maximum a posteriori path
% prec - bands of the precision matrix, which is tridiagonal (or block
%   tridiagonal for higher dimension state spaces) due to conditional
%   independence properties of state space models
% ll - log likelihood of the latent path at the end of the E step
% David Pfau, 2012

opts = struct( 'GradObj', 'on', 'Display', 'off', 'LargeScale', 'on', 'Hessian', 'on', 'HessMult', @hess_mult );
map = init;
[map,ll,~,~,~,prec] = fminunc(@(x) log_lik( data, x, params ), map, opts );