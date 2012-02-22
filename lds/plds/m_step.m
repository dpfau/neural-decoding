function [params ecll fe] = m_step( data, map, prec, params )
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

Ptt1  = (sum(covar.off_diag,3) + map(:,1:end-1)*map(:,2:end)')';
Pt1t1 = sum(covar.diag(:,:,1:end-1),3) + map(:,1:end-1)*map(:,1:end-1)';
Ptt   = sum(covar.diag(:,:,2:end),3) + map(:,2:end)*map(:,2:end)';

A = Ptt1/Pt1t1;
Q = 1/(T-1)*(Ptt - A*Ptt1'); % This part is nearly identical to the standard LDS case

aug_covar = [ covar.diag, zeros(k,1,T); zeros(1,k+1,T) ]; % Augment covariance with zeros for bias term
opts = optimset('GradObj','on','Display','off');
[Cb dat_ll] = fminunc( @(x) data_ll( x, data, [map; ones(1,size(map,2))], aug_covar ), [params.C, params.b], opts ); % Also augment mean with row of ones for bias term, and estimate C and b simultaneously by numerical optimization

params = struct( 'A', A, 'C', Cb(:,1:end-1), 'Q', Q, 'b', Cb(:,end), 'f', params.f );
Qinv = Q^-1;
ecll = dat_ll + 1/2*sum( sum( Qinv*Pt1t1 - 2*Qinv*A*Ptt1 + A'*Qinv*A*Ptt ) );
fe = ecll - entropy( prec ); 

function [f grad] = data_ll( C, data, map, covar )

Cmu = C*map;
sigCt = tprod( covar, [1 -1 3], C, [2 -1] );
CsigCt = tprod( sigCt, [-1 1 2], C, [1 -1] );
fCx = exp( Cmu + 1/2*CsigCt );

f = sum( sum( fCx - data.*Cmu ) );
grad = fCx*map' + tprod( fCx, [1 -1], sigCt, [2 1 -1] ) ...
     - data*map';

% function test_data_ll( C, data, map, covar )
%         
% [fx,grad] = data_ll( C, data, map, covar );
% for i = 1:numel(C)
%     C(i) = C(i) + 1e-8;
%     fx_ = data_ll( C, data, map, covar );
%     fprintf('Exact gradient: %d, Approximate gradient: %d\n',grad(i),(fx_-fx)*1e8);
%     C(i) = C(i) - 1e-8;
% end
 
% function f = mc_data_ll( C, data, map, covar, t )
% % Instead of calculating data part of expected complete log likelihood
% % directly, sample Monte Carlo estimate
% 
% k = size(covar,1);
% N = size(covar,3);
% Q = zeros(k,k*N);
% for i = 1:N
%     Q(:,(i-1)*k + (1:k)) = chol(covar(:,:,i));
% end
% 
% f = zeros(t,1);
% for i = 1:t
%     z = map + reshape( Q*randn(numel(map),1), k, N );
%     f(i) = sum( sum( exp( C*[z; ones(1,size(z,2))] ) - data.*C*[z; ones(1,size(z,2))] ) );
% end
        