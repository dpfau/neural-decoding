function [y x] = gen( params, len, x0 )

x = zeros( size( params.A, 1 ), len );

xt = x0;
R = chol(params.Q);
for i = 1:len
    x(:,i) = xt;
    xt = params.A*xt + R*randn(size(R,2),1);
end
y = poissrnd( params.f( add_vector( params.C*x, params.b ), 0 ) );