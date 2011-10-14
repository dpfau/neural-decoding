function [A B C D x0 s] = moesp( y, u, i, N, proj, eps )
% y - output data, one column per time step
% u - input data, one column per time step
% i - number of block-Hankel rows.  i*l should be greater than system order
% N - number of timesteps from the data used in reconstruction
% proj - the type of projection used: orth_svd, orth_pinv or oblique
% eps - the ratio between the greatest singular value and the last one
%   used for choosing the system order, or, if it's a negative number, -1
%   times the system order.
% David Pfau, 2011

l = size( y, 1 );
m = size( u, 1 );

Y = block_hankel( y, 1, 2*i, N );
U = block_hankel( u, 1, 2*i, N );

Yf = Y(i+1:end,:);
Uf = U(i+1:end,:);

% Oi = Y * ( eye( N ) - pinv( U ) * U );
Oi = oblique( Yf, Uf, [Y(1:i,:);U(1:i,:)] );

[r,s,~] = svd( Oi );
n = find( diag( s ) < eps, 1 ) - 1; % approximate order of the system
j = i*l - n;
G = r( :, 1:n ) * sqrt( s( 1:n, 1:n ) );
C = G( 1:l, : );
% A = pinv( G( 1:(i-1)*l, : ) ) * G( (l+1):i*l, : );
A = pinv( G ) * [ G( (l+1):i*l, : ); zeros( l, n ) ]; % stable A trick
Gperp = r( :, n+1:end )';
M = Gperp * Yf * pinv( Uf );
L = zeros( i*j, size( Gperp, 2 ) );
M1 = zeros( i*j, m );

for k = 0:i-1
    L( k * j + (1:j), : ) = [ Gperp( :, k*l+1:end ), zeros( j, k*l ) ];
    M1( k * j + (1:j), : ) = M( :, k * m + (1:m) ); 
end
DB = pinv( L * [ eye(l), zeros(l,n); zeros((i-1)*l,l), G( 1:(i-1)*l, : ) ] ) * M1;
D = DB( 1:l, : );
B = DB( l+1:end, : );

h = [D; zeros((i-1)*l,m)];
for k = 0:i-2
    h(k*l + (1:l),:) = C*A^k*B;
end
H = [h, zeros(i*l,(i-1)*m)];
for k = 1:i-1
    H(:,k*m + (1:m)) = [zeros(k*l,m); H(1:end-k*l,1:m)];
end
x0 = pinv( G ) * ( Y( 1:i, 1 ) - H * U( 1:i, 1 ) );
s = diag(s);