function params = m_step( data, map, prec, params )
% Recovers maximum expected complete likelihood parameters of Poisson-LDS
% model given data, the MAP path over time and the Hessian of the complete 
% log likelihood, which is the precision matrix of the best Gaussian 
% approximation to the posterior distribution over latent position
%
% data - the observed data
% map - the MAP latent position over time
% prec - a struct containing the information needed to reconstruct the
%   precision matrix
% params - estimated parameters that maximize the expected complete log
%   likelihood
% ecll - the (negative) expected complete log likelihood
% fe - the free energy, or ECLL plus the entropy of the path posterior
%
% David Pfau, 2012

T = size(map,2);
k = size(map,1);
N = size(data,1);
t = params.t;
covar = inv_block_tridiag( prec );
params.x0 = map(:,1);
params.Q0 = covar.diag(:,:,1) + map(:,1)*map(:,1)';

Ptt1  = (sum(covar.off_diag,3) + map(:,1:end-1)*map(:,2:end)')';
Pt1t1 = sum(covar.diag(:,:,1:end-1),3) + map(:,1:end-1)*map(:,1:end-1)';
Ptt   = sum(covar.diag(:,:,2:end),3) + map(:,2:end)*map(:,2:end)';

params.A = Ptt1/Pt1t1;
params.Q = (Ptt - params.A*Ptt1')/(T-1); % This part is nearly identical to the standard LDS case
params.Q = 1/2*(params.Q + params.Q'); % Enforce symmetry.

opts = optimset('GradObj','on','Display','iter');
warning('off','MATLAB:nearlySingularMatrix')
CbD = fminunc( @(x) data_ll( x, ...
                             data, ...
                             [map; ones(1,size(map,2));zeros(N*t,t), block_hankel(data,1,t,T-1)],...
                             covar.diag, ...
                             k), ...
               [params.C, params.b, params.D], ...
               opts ); % Also augment mean with row of ones for bias term, and estimate C and b simultaneously by numerical optimization
warning('on','MATLAB:nearlySingularMatrix')

params.C = CbD(:,1:k);
params.b = CbD(:,k+1);
params.D = CbD(:,k+2:end);

function [f grad Hinfo] = data_ll( C, data, map, covar, dim )

x = size(C,1);
y = size(C,2);
Cmu = C*map;
sigCt = tprod( covar, [1 -1 3], C(:,1:dim), [2 -1] );
CsigCt = tprod( sigCt, [-1 1 2], C(:,1:dim), [1 -1] );
fCx = exp( Cmu + 1/2*CsigCt );

f = sum( sum( fCx - data.*Cmu ) );
grad = fCx*map' ...
     + [tprod( fCx, [1 -1], sigCt, [2 1 -1] ), zeros(x,y-dim)] ...
     - data*map';
% mapSigCt = tprod(map,[1 3],ones(size(sigCt,2),1),2)+cat(1,sigCt,zeros(size(map,1)-size(sigCt,1),size(sigCt,2),size(sigCt,3)));
% H =  tprod( fCx, [1 -1], covar, [2 3 -1] ) + tprod( fCx, [1 -1], tprod(mapSigCt,[1 2 4],mapSigCt,[1 3 4]), [1 2 3 -1] );
H1 =  tprod( fCx, [1 -1], covar, [2 3 -1] ) ...
    + tprod( fCx, [1 -1], tprod( map(1:dim,:), [1 4], sigCt, [2 3 4] ) ...
                        + tprod( sigCt, [1 3 4], map(1:dim,:), [2 4] ) ...
                        + tprod( sigCt, [1 3 4], sigCt, [2 3 4] ), [2 3 1 -1] );
H2 = cat(3,cat(2,H1,zeros(x,y-dim,dim)),zeros(x,y,y-dim)) ...
   + tprod( fCx, [1 -1], tprod( map, [1 3], map, [2 3]), [2 3 -1] );
Hinfo = struct('all',H2);

function test_data_ll( C, data, map, covar, dim )

[fx,grad] = data_ll( C, data, map, covar, dim );
for i = 1:numel(C)
    C(i) = C(i) + 1e-5;
    fx_ = data_ll( C, data, map, covar, dim );
    grad_ = (fx_-fx)*1e5;
    fprintf('Exact: %d, Approx: %d, Ratio: %d\n',grad(i),grad_,(grad(i)-grad_)/grad(i));
    C(i) = C(i) - 1e-5;
end   