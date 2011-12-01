function ku = orth_kernel(k,U)
% given a kernel k(x,x') and a matrix U, construct a kernel ku that is the
% inner product in the orthogonal projection of U in feature space.  That
% is, ku(x,u) = 0 for all x and u such that u is a column of U.

Kuu = zeros(size(U,2));
for i = 1:size(U,2)
    for j = 1:size(U,2)
        Kuu(i,j) = k(U(:,i),U(:,j));
    end
end
invKuu = Kuu^-1;
ku = @(x,y) k(x,y) - kux(x,U,k)'*invKuu*kux(y,U,k);
kuf = @kux;

function z = kux(x,U,k)
z = zeros(size(U,2),1);
for i = 1:size(U,2)
    z(i) = k(x,U(:,i));
end