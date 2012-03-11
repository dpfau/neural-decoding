function ent = entropy( prec )
% In keeping with our convention of using negative log likelihood, return
% the *negative* entropy of a Gaussian given the precision matrix

if isfield(prec,'diag')
    N = length(prec.diag);
elseif isfield(prec,'diag_center')
    N = size(prec.diag_center,2)*size(prec.diag_left,1);
else
    error('Not a recognized format for precision matrix')
end
ent = 1/2*( N*log(2*pi) + N + log_det_tridiag( prec ) );