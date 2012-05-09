function y = log_det_tridiag( x )
% Log-Determinant of a tridiagonal matrix.  If the diagonal is a and the
% off-diagonal is b, then the determinant is given by the final element of
% the sequence:
% y(0) = 1
% y(1) = a(1)
% y(n) = a(n)*y(n-1) - b(n-1)^2*y(n-2)
% Since this quickly diverges if n is large, we work in the log domain as
% much as possible
% David Pfau, 2012

if isfield(x,'diag')
    z = 0;
    y = log(x.diag(1));
    for i = 2:length(x.diag)
        w = x.diag(i)*exp(y - z) - x.off_diag(i-1)^2;
        foo = y;
        y = log(w)+z;
        z = foo;
    end
else
    y = 2*sum(log(diag(cholreg(sparse_hess(x),5e-3))));
end