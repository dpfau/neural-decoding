function [params ecll fe] = m_step( data, map, prec, f )
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
covar = inv_block_tridiag( prec );

Ptt1  = sum(covar.off_diag,3) + map(:,1:end-1)*map(:,2:end)';
Pt1t1 = sum(covar.diag(:,:,1:end-1),3) + map(:,1:end-1)*map(:,1:end-1)';
Ptt   = sum(covar.diag(:,:,2:end),3) + map(:,2:end)*map(:,2:end)';

A = (Ptt1/Pt1t1)';
Q = 1/(T-1)*(Ptt - A*Ptt1'); % This part is nearly identical to the standard LDS case

aug_covar = [ covar.diag, zeros(k,1,N); zeros(1,k+1,N) ]; % Augment covariance with zeros for bias term
Cb = fminunc( @(x) data_ll( x, data, [map; ones(1,size(map,2))], aug_covar ) ); % Also augment mean with row of ones for bias term, and estimate C and b simultaneously by numerical optimization
params = struct('A',A,'C',Cb(:,1:end-1),'Q',Q,'b',Cb(:,end),'f',f);

function [f grad] = data_ll( C, data, map, covar )

Cmu = C*map;
sigCt = tprod( covar.diag, [1 -1 3], C, [2 -1] );
CsigCt = tprod( sigCt, [-1 1 2], C, [1 -1] );

f = sum( sum( exp( Cmu + 1/2*CsigCt ) - data.*Cmu ) );
grad = tprod( exp( Cmu + 1/2*CsigCt ), [1 -1], tprod( map, [1 3], ones(1,size(C,1)), [1 2] ) + sigCt, [2 1 -1] ) ...
     - tprod( data, [1 -1], map, [2 -1] );