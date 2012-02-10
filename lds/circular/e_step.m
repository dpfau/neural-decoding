function [map prec ll] = e_step( data, params, init )
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
% delta - the change between the initial and final path
% David Pfau, 2012

opts = struct( 'GradObj', 'on', 'Display', 'off', 'LargeScale', 'on', 'Hessian', 'on', 'HessMult', @hess_mult );
if nargin < 3
    init = 0:params.dt:params.dt*(length(data)-1); % Initialize with no variance in phase shift
end
map = init;
[map,ll,~,~,~,prec] = fminunc(@(x) log_lik( data, x, params ), map, opts );

% for t = 1:10
%     [~,grad,Hinfo] = log_lik( data, map, params );
%     v = randn(size(grad));
%     hv = hess_mult( Hinfo, v' );
%     map = map + v*1e-8;
%     [~,grad2,~] = log_lik( data, map, params );
%     map = map - v*1e-8;
%     hv2 = (grad2-grad)/1e-8;
%     fprintf('Exact value: %d, Approx value: %d, Difference: %d\n',norm(hv),norm(hv2),norm(hv'-hv2));
% end

% function test_d2sinct( map, params )
% 
% T = length(params.template);
% coord = interp_coord( map, T );
% d1 = sinct(coord,1);
% 
% map = map + 1e-8;
% coord2 = interp_coord( map, T );
% d2 = sinct(coord2,1);
% 
% dd1 = (d2-d1)/1e-8;
% dd2 = sinct(coord,2);
% pause(1)


function hv = hess_mult( Hinfo, v )
% Multiplies the vector v by the Hessian of the log likelihood, with
% sufficient statistics in the struct Hinfo

hv = zeros(size(v));
for i = 1:size(v,2)
    hv(:,i) = Hinfo.diag'.*v(:,i) + [0;Hinfo.off_diag'.*v(1:end-1,i)] + [Hinfo.off_diag'.*v(2:end,i);0];
end