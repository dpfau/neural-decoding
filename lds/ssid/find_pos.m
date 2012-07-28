function x = find_pos( A )
% For an overdetermined linear function A, find x such that every element
% of A*x is strictly positive (if possible)

x = zeros( size( A, 2 ), 1 );
c = 1;
while nnz( A*x <= 0 ) > 0
    c = c*2;
    x = fminunc( @(x) log_reg( A, x, c ), x, struct('GradObj','on','Display','iter') );
    % x = newtons_method( @(x) log_reg( A, x, c ), x, 1e-16, 1 );
    fprintf( '%d negative or zero entries\n', nnz( A*x <= 0 ) );
end

function [y grad] = log_reg( A, x, c )

% [y_log grad_log hess_log] = logistic( c*A*x );
[y_log grad_log] = logistic( c*A*x );
y = sum( y_log ) + 0.5*x'*x/c;
grad = sum( c*A.*( grad_log*ones( 1, size( A, 2 ) ) ), 1 )' + x/c;
% hess = tprod( tprod( c*A, [3 1], c*A, [3 2] ), [1 2 -1], hess_log, -1 ) + eye( numel( x ) )/c;

function [y grad] = logistic( x )

ex  = exp( x );
ex_ = exp( -x );
y    = -1./(1+ex_);
grad = -1./(2+ex_+ex);
% hess = (ex-ex_)./(2+ex_+ex).^2;