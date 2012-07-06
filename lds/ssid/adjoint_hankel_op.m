function y = adjoint_hankel_op( Un, m, i, N, s, x, mode )

switch mode
    case 0
        y = { [ m * i, size( Un, 2 ) ], [ m, N + s ] };
    case 1
        y = [ adjoint_hankel( x * Un', i, N ), zeros( m, s ) ]; 
    case 2
        y = block_hankel( x(:,1:end-s), 1, i, N ) * Un;
    otherwise
        error( 'Not a recongized mode for TFOCS linear operator' )
end