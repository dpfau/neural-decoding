function A = stabilize( A )
% one technique for stabilizing an unstable state transition matrix,
% pretty much a copy of the version in Zhang Liu's code.  David Pfau, 2011

n = size(A,1);
[UA,TA] = schur( A, 'complex' );
eigs = diag(TA);
ns = nnz( abs( eigs ) > 1 );
while ns > 0
    eigs( abs( eigs ) > 1 ) = 1./eigs( abs( eigs ) > 1 );
    TA( 1:n+1:end ) = eigs;
    A = real( UA*TA*UA' );
    [UA,TA] = schur( A, 'complex' );
    eigs = diag(TA);
    ns = nnz( abs( eigs ) > 1 );
end