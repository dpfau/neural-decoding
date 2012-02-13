function [E_xt_xt E_xt_xt1 covar] = prod_expectation( map, prec )
% Given the mean and precision of a random vector X, returns the marginal
% expectations of X(i)^2 and X(i)*X(i+1), assuming X is a Markov chain
%
% Input:
%   map - E[X]
%   prec - precision matrix of X, which is tridiagonal given the Markov
%   constraint.  Given as a struct that contains two fields: diag, the
%   diagonal, and off_diag, the off diagonal
%
% Output:
%   E_xt_xt - E[X_t^2]
%   E_xt_xt1 - E[X_t*X_t+1]

covar = inv_tridiag( prec );
E_xt_xt  = covar.diag + map.^2;
E_xt_xt1 = covar.off_diag + map(1:end-1).*map(2:end);