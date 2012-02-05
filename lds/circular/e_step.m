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

%opts = struct( 'GradObj', 'on', 'Display', 'iter', 'LargeScale', 'on', 'Hessian', 'on', 'HessMult', @hess_mult );
map = 0:params.dt:params.dt*(length(data)-1); % Initialize with no variance in phase shift
%map = fminunc(@(x) log_lik( data, x, params ), map, opts );
for t = 1:10
    [~,grad,Hinfo] = log_lik( data, map, params );
    v = randn(size(grad));
    hv = hess_mult( Hinfo, v );
    map = map + v*1e-8;
    [~,grad2,~] = log_lik( data, map, params );
    map = map - v*1e-8;
    hv2 = (grad2-grad)/1e-8;
    fprintf('Exact norm: %d, Approx norm: %d, Diff norm: %d\n',norm(hv),norm(hv2),norm(hv-hv2));
end

function [ll grad Hinfo] = log_lik( data, x, params )

coord = interp_coord( x, length(params.template) );
ll = sum( -( interp( coord, params.template ) - data ).^2/(2*params.sig_temp^2) ) + ...
    sum( -( (x(2:end)-x(1:end-1)) - params.dt ).^2/(2*params.sig_dt^2) ) + ...
    -( x(1) - params.t0 )^2/(2*params.sig_dt^2);

grad = zeros(size(x));
grad(1) = (params.t0 - x(1))/params.sig_dt^2;
grad(1:end-1) = grad(1:end-1) - ( x(1:end-1) + params.dt - x(2:end) )/params.sig_dt^2;
grad(2:end)   = grad(2:end)   - ( x(2:end) - params.dt - x(1:end-1) )/params.sig_dt^2;
grad = grad + ( data - interp( coord, params.template ) ).*dinterp( coord, params.template )/params.sig_temp^2;

Hinfo = struct('diag',-2/params.sig_dt^2*ones(size(grad)),'off_diag',ones(1,size(grad,2)-1)/params.sig_dt^2);
Hinfo.diag(end) = -1/params.sig_dt^2;
Hinfo = struct('diag',zeros(size(grad)),'off_diag',zeros(1,size(grad,2)-1));
Hinfo.diag = Hinfo.diag + ( -(dinterp( coord, params.template ).^2) + ...
    ( data - interp( coord, params.template ) ).*d2interp( coord, params.template ) )/params.sig_temp^2;

ll = -ll; grad = -grad; % Hack because I want to maximize, not minimize.
Hinfo.diag = -Hinfo.diag; Hinfo.off_diag = -Hinfo.off_diag;

function hv = hess_mult( Hinfo, v )
% Multiplies the vector v by the Hessian of the log likelihood, with
% sufficient statistics in the struct Hinfo

hv = Hinfo.diag.*v + [0,Hinfo.off_diag.*v(1:end-1)] + [Hinfo.off_diag.*v(2:end),0];

function y = interp( coord, template ) 
% Whittaker-Shannon interpolation

y = template'*sinc(coord); 

function y = dinterp( coord, template ) 
% derivative of Whittaker-Shannon interpolation wrt time at which we are interpolating

y = template'*dsinc(coord)*length(template)/2/pi;
y(isnan(y)) = 0;

function y = d2interp( coord, template )
% second derivative of Whittaker-Shannon interpolation

y = template'*d2sinc(coord)*length(template)^2/4/pi^2;
y(isnan(y)) = 0;

function coord = interp_coord( x, N ) 
% Given continuous time points and the period of a signal, returns matrix 
% of coordinates for reconstructing signal at those time points using
% Whittaker-Shannon interpolation

phase = mod( x, 2*pi )*N/(2*pi); % map all points on the path into [0,length(template))
coord = ones(N,1)*phase - (1:N)'*ones(1,size(phase,2));
coord( coord > N/2 )  = coord( coord > N/2 )  - N; % shift difference between phase and template points from (-2*pi,2*pi) to (-pi,pi)
coord( coord < -N/2 ) = coord( coord < -N/2 ) + N;

function y = dsinc( x ) % derivative of sin(pi*x)/(pi*x) wrt x

y = cos(pi*x)./x - sin(pi*x)./x.^2/pi;

function y = d2sinc( x ) % second derivative of sinc(x) = sin(pi*x)/(pi*x)

y = 2*sin(pi*x)./x.^3/pi - 2*cos(pi*x)./x.^2 - pi*sin(pi*x)./x;