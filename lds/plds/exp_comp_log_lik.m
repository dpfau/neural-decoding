function ecll = exp_comp_log_lik( data, map, prec, params )
% The expected complete log likelihood given data, Gaussian approximation
% to latent state posterior, and parameters for a Poisson-LDS model

k = size(map,1);
T = size(map,2);

covar = inv_block_tridiag( prec );

Qinv = params.Q^-1;
A = params.A;
C = [params.C params.b];

Ptt1  = sum(covar.off_diag,3) + map(:,1:end-1)*map(:,2:end)';
Pt1t1 = sum(covar.diag(:,:,1:end-1),3) + map(:,1:end-1)*map(:,1:end-1)';
Ptt   = sum(covar.diag(:,:,2:end),3) + map(:,2:end)*map(:,2:end)';

aug_covar = [ covar.diag, zeros(k,1,T); zeros(1,k+1,T) ];
Cmu = params.C*map;
sigCt = tprod( aug_covar, [1 -1 3], C, [2 -1] );
CsigCt = tprod( sigCt, [-1 1 2], C, [1 -1] );

ecll = sum( sum( exp( Cmu + 1/2*CsigCt ) - data.*Cmu ) ) ...
       + 1/2*sum( sum( Qinv*Pt1t1 - 2*Qinv*A*Ptt1 + A'*Qinv*A*Ptt ) );