function [A B C D X Q R S] = n4sid( y, u, i, eps )

m = size(u,1);
l = size(y,1);
Y = block_hankel( y, 1, 2*i, size(y,2) - 2*i + 1 );
U = block_hankel( u, 1, 2*i, size(u,2) - 2*i + 1 );

Yp = Y( 1:i*l, : );
Yf = Y( i*l+1:end, : );
Up = U( 1:i*m, : );
Uf = U( i*m+1:end, : );

Yp1 = Y( 1:(i+1)*l, : );
Yf1 = Y( (i+1)*l+1:end, : );
Up1 = U( 1:(i+1)*m, : );
Uf1 = U( (i+1)*m+1:end, : );

Oi  = oblique( Yf, Uf, [Yp; Up] );
Oi1 = oblique( Yf1, Uf1, [Yp1; Up1] );

[r,s,v] = svd(Oi);
n = find( diag( s ) < eps, 1 ) - 1; % approximate order of the system
G = r( :, 1:n ) * sqrt( s( 1:n, 1:n ) ); % Extended observability matrix

X  = sqrt( s( 1:n, 1:n ) ) * v( :, 1:n )';
X1 = pinv( G( 1:end-l, : ) ) * Oi1;

Uii = u(:,i+(1:size(u,2) - 2*i + 1));
Yii = y(:,i+(1:size(y,2) - 2*i + 1));

ABCD = [X1; Yii] * pinv( [X; Uii] );
A = ABCD( 1:n, 1:n );
B = ABCD( 1:n, n+1:end );
C = ABCD( n+1:end, 1:n );
D = ABCD( n+1:end, n+1:end );

resid = [X1; Yii] - ABCD*[X; Uii];
covar = resid * resid';
Q = covar( 1:n, 1:n );
R = covar( n+1:end, n+1:end );
S = covar( 1:n, n+1:end );