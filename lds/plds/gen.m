function [x y] = gen( params, len, x0 )

y = zeros( size( params.A, 1 ), len );

yt = x0;
R = chol(params.Q);
for i = 1:len
    y(:,i) = yt;
    yt = params.A*yt + R*randn(size(R,2),1);
end
x = poissrnd( params.f( add_vector( params.C*y, params.b ), 0 ) );