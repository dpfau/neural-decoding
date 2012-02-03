function [map prec] = e_step( data, params )
% Given data and parameters for a latent-phase model, find the Laplace
% approximation to the expected log likelihood by first finding the MAP
% path, then taking a quadratic approximation of the log likelihood around
% that path.
%
% Input:
% data - vector of observed data
% params - struct of parameters in latent-phase model, with fields:
%   t0 - initial phase
%   dt - the average rate of phase advancing
%   sig_dt - the variance in the phase advance at each time step
%   template - vector of average observations over one cycle
%   sig_temp - variance around each observation
% 
% Output:
% map - maximum a posteriori path
% prec - bands of the precision matrix, which is tridiagonal (or block
%   tridiagonal for higher dimension state spaces) due to conditional
%   independence properties of state space models
% David Pfau, 2012

opts = struct( 'GradObj', 'on', 'LargeScale', 'on' );
map = 0:params.dt:params.dt*(length(data)-1); % Initialize with no variance in phase shift
map = fminunc(@(x) log_lik( data, x, params ), map, opts );

function [ll grad] = log_lik( data, x, params )

coord = interp_coord( x, length(params.template) );
ll = sum( -( interp( coord, params.template ) - data ).^2/(2*params.sig_temp^2) ) + ...
    sum( -( (x(2:end)-x(1:end-1)) - params.dt ).^2/(2*params.sig_dt^2) ) + ...
    -( x(1) - params.t0 )^2/(2*params.sig_dt^2);

grad = zeros(size(x));
grad(1) = (params.t0 - x(1))/params.sig_dt^2;
grad(1:end-1) = grad(1:end-1) - ( x(1:end-1) + params.dt - x(2:end) )/params.sig_dt^2;
grad(2:end)   = grad(2:end)   - ( x(2:end) - params.dt - x(1:end-1) )/params.sig_dt^2;
grad = grad + ( data - interp( coord, params.template ) ).*dinterp( coord, params.template )/params.sig_temp^2;

function y = interp( coord, template )

y = template'*sinc(coord) ; % Whittaker-Shannon interpolation

function y = dinterp( coord, template ) % derivative of Whittaker-Shannon interpolation wrt time at which we are interpolating

y = template'*(cos(pi*coord)./coord - sin(pi*coord)./coord.^2/pi)*length(template)/(2*pi);

function coord = interp_coord( x, N )

phase = mod( x, 2*pi )*N/(2*pi); % map all points on the path into [0,length(template))
coord = ones(N,1)*phase - (1:N)'*ones(1,size(phase,2));
coord( coord > N/2 )  = coord( coord > N/2 )  - N; % shift difference between phase and template points from (-2*pi,2*pi) to (-pi,pi)
coord( coord < -N/2 ) = coord( coord < -N/2 ) + N;