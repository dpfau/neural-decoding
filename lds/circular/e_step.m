function [map prec] = e_step( data, params, init )
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
% init - initial value for phase variables
% 
% Output:
% map - maximum a posteriori path
% prec - bands of the precision matrix, which is tridiagonal (or block
%   tridiagonal for higher dimension state spaces) due to conditional
%   independence properties of state space models
% David Pfau, 2012

opts = struct( 'GradObj', 'on', 'Display', 'iter', 'LargeScale', 'on', 'Hessian', 'on', 'HessMult', @hess_mult );
if nargin < 3
    map = 0:params.dt:params.dt*(length(data)-1); % Initialize with no variance in phase shift
else
    map = init;
end
map = fminunc(@(x) log_lik( data, x, params ), map, opts );
[~,~,prec] = log_lik( data, map, params );
% for t = 1:10
%     [~,grad,Hinfo] = log_lik( data, map, params );
%     v = randn(size(grad));
%     hv = hess_mult( Hinfo, v );
%     map = map + v*1e-8;
%     [~,grad2,~] = log_lik( data, map, params );
%     map = map - v*1e-8;
%     hv2 = (grad2-grad)/1e-8;
%     fprintf('Exact value: %d, Approx value: %d, Difference: %d\n',norm(hv),norm(hv2),norm(hv-hv2));
% end

% function test_d2sinct( map, params )
% 
% T = length(params.template);
% coord = interp_coord( map, T );
% d1 = dsinct(coord);
% 
% map = map + 1e-8;
% coord2 = interp_coord( map, T );
% d2 = dsinct(coord2);
% 
% dd1 = (d2-d1)/1e-8;
% dd2 = d2sinct(coord);
% pause(1)

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
Hinfo.diag = Hinfo.diag + ( -(dinterp( coord, params.template ).^2) + ...
    ( data - interp( coord, params.template ) ).*d2interp( coord, params.template ) )/params.sig_temp^2;

ll = -ll; 
grad = -grad; % Hack because I want to maximize, not minimize.
Hinfo.diag = -Hinfo.diag; 
Hinfo.off_diag = -Hinfo.off_diag;

function hv = hess_mult( Hinfo, v )
% Multiplies the vector v by the Hessian of the log likelihood, with
% sufficient statistics in the struct Hinfo

hv = zeros(size(v));
for i = 1:size(v,2)
    hv(:,i) = Hinfo.diag'.*v(:,i) + [0;Hinfo.off_diag'.*v(1:end-1,i)] + [Hinfo.off_diag'.*v(2:end,i);0];
end

function y = interp( coord, template ) 
% Whittaker-Shannon interpolation, approximately.  Instead of using
% sin(pi*x)/(pi*x), use sin(pi*x)/(T*tan(pi*x/T)).  Not the exact 
% interpolation formula (terms go to zero too quickly far away from the 
% center), but close enough for our purposes.

y = template'*sinct(coord); 

function y = dinterp( coord, template ) 
% derivative of Whittaker-Shannon interpolation wrt time at which we are interpolating

y = template'*dsinct(coord)*length(template)/2/pi;

function y = d2interp( coord, template )
% second derivative of Whittaker-Shannon interpolation

y = template'*d2sinct(coord)*length(template)^2/4/pi^2;

function coord = interp_coord( x, N ) 
% Given continuous time points and the period of a signal, returns matrix 
% of coordinates for reconstructing signal at those time points using
% Whittaker-Shannon interpolation

coord = ones(N,1)*x*N/(2*pi) - (1:N)'*ones(1,size(x,2));

function y = sinct( x )

T = size(x,1);
y = sin(pi*x).*cot(pi*x/T)/T;
y(isnan(y)) = 1;

function y = dsinct( x ) % derivative of sin(pi*x)/(T*tan(pi*x/T)) wrt x

T = size(x,1);
y = pi/T  *cos(pi*x).*cot(pi*x/T) - ...
    pi/T^2*sin(pi*x).*csc(pi*x/T).^2;
y(isnan(y)) = 0; % deal with the singularity at 0

function y = d2sinct( x ) % second derivative of sinct(x,T) = sin(pi*x)/(T*tan(pi*x/T))

T = size(x,1);
y = -2*pi^2/T^2*cos(pi*x).*csc(pi*x/T).^2 - ...
    pi^2/T     *sin(pi*x).*cot(pi*x/T)    + ...
    2*pi^2/T^3 *sin(pi*x).*cot(pi*x/T).*csc(pi*x/T).^2;
y(isnan(y)) = -pi;