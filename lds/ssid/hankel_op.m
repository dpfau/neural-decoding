function y = hankel_op( Un, m, i, N, x, mode )
% Given x, calculates X*Un, where X is a block-Hankel matrix derived from
% x, and Un is a matrix whose columns span the null space of the
% block-Hankel matrix of u.  Standard form of linear operators in TFOCS.

switch mode
    case 0 % return { input size, output size }
        y = { [ m, N ], [ m * i, size( Un, 2 ) ] };
    case 1 % apply operator to input
        y = block_hankel( x, 1, i, N ) * Un;
    case 2 % apply adjoint operator to input
        y = adjoint_hankel( x * Un', i, N );
end