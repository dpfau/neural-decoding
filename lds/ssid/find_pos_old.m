function x = find_pos_old( A )
% For an overdetermined linear function A, find x such that every element
% of A*x is strictly positive

x = zeros( size( A, 2 ), 1 );
c = 1;
while nnz( x <= 0 ) > 0
    c = c*2;
    x = fminunc( @(x) log_reg( A, x, c ), x, struct('GradObj','on','Display','iter') );
    fprintf( '%d negative or zero entries', nnz( x <= 0 ) );
end

function [y grad] = log_reg( A, x, c )

[y_log grad_log] = logistic( c*A*x );
y = sum( y_log ) + 0.5*x'*x/c;
grad = sum( c*A.*( grad_log*ones( 1, size( A, 2 ) ) ), 1 )' + x/c;

function [y grad] = logistic( x )

y    = -1./(1+exp(-x));
grad = -1./(2+exp(-x)+exp(x));