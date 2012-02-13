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
T = length(data);
t0 = map(1);
dt = mean(diff(map));
if nargin < 4 % If we only have the mean of the posterior path probability
    sig_dt = std(diff(map));
    coord = interp_coord( map, N );
    template = data*pinv( sinct( coord, 0 ) ); % Since the template is a linear function of the coordinates, just minimize the mean squared error
    sig_temp = std( data - template*sinct( coord, 0 ) );
else % With both the mean and covariance of the posterior path probability we have sufficient statistics for the M-step 
    [extt extt1 covar] = prod_expectation( map, prec );
    sig_dt = sqrt( ( -extt(1) - extt(end) - 2*sum(extt(2:end-1)) + 2*map(end)*dt - 2*map(1)*dt + 2*sum(extt1) - (T-1)*dt^2  )/( 1 - T ) );
    coord = interp_coord( map, N );
    sinct0 = sinct( coord, 0 );
    sinct1 = sinct( coord, 1 ).*(ones(size(coord,1),1)*covar.diag)*N/2/pi;
    sinct2 = sinct( coord, 2 ).*(ones(size(coord,1),1)*covar.diag.^2)*N^2/4/pi^2;
    Y = (sinct0 + 1/2*sinct2)*data'; % sum_t data(t) * E[sinct(coord,0)], using quadratic approximation for E[sinct(coord,0)]
    F = (sinct0*sinct0') + (sinct1*sinct1') + 1/2*(sinct0*sinct2') + 1/2*(sinct2*sinct0') + 3/4*(sinct2*sinct2'); % sum_t E[sinct(coord,0)*sinct(coord,0)']
    template = Y'*pinv( F ); % All experimental evidence shows this is overthinking it and doesn't actually affect the result...maybe
    sig_temp = sqrt( ( template*F*template' -2*template*Y + data*data' )/T );
end

params = struct('t0',t0,'dt',dt,'sig_dt',sig_dt,'sig_temp',sig_temp,'template',template);