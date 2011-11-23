function X = oblique(A,B,C)
% the oblique projection of A onto C along B

assert( size(A,2) == size(B,2) );
assert( size(A,2) == size(C,2) );

Bperp = orthog(B);
X = (A*Bperp)*pinv(C*Bperp)*C;

function Bperp = orthog(B)

Bperp = eye(size(B,2)) - pinv(B)*B;