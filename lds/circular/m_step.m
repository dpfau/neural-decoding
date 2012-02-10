function params = m_step( data, N, map, prec )
% Recovers maximum expected complete likelihood parameters of latent-phase
% model given data, number of points in a template, the MAP phase over
% time, and if available, the Hessian of the complete log likelihood, which
% is the precision matrix of the best Gaussian approximation to the
% posterior distribution over phases
%
% data - the observed data
% N - the number of points in a template
% map - the MAP phase over time
% prec - a struct with two fields, diag and off_diag, which are the
%   diagonal and off-diagonal of the precision matrix of the path
%   posterior, which is tridiagonal due to the conditional independence of
%   non-adjacent points
%
% David Pfau, 2012

assert( length(map) == length(data), 'Phase path and data are not the same length' )
t0 = map(1);
dt = mean(diff(map));
if nargin < 4 % If we only have the mean of the posterior path probability
    sig_dt = std(diff(map));
    coord = ones(N,1)*map*N/(2*pi) - (1:N)'*ones(1,size(map,2));
    template = data*pinv( sinct( coord, 0 ) ); % Since the template is a linear function of the coordinates, just minimize the mean squared error
    sig_temp = std( data - template*sinct( coord, 0 ) );
else % With both the mean and covariance of the posterior path probability we have sufficient statistics for the M-step 
end

params = struct('t0',t0,'dt',dt,'sig_dt',sig_dt,'sig_temp',sig_temp,'template',template);