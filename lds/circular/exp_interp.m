function [ex ex2] = exp_interp( map, prec, template )
% If X ~ N(map,sig), calculate the expectation of interp( map, template )
% and interp( map, template )^2 by quadratic approximation of interp around
% map.

covar = inv_tridiag( prec );
in0 = interp( map, template, 0 );
in1 = interp( map, template, 1 );
in2 = interp( map, template, 2 );
ex = in0 + 1/2*in2.*covar.diag;
ex2 = in0.^2 + (in1.^2 + in0.*in2).*covar.diag + 3/4*in2.^2.*covar.diag.^2;
