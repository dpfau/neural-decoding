function ent = entropy( prec )
% Gives the entropy of a Gaussian with tridiagonal precision matrix prec

N = length(prec.diag);
ent = -1/2*( N*log(2*pi) + N + log_det_tridiag( prec ) );