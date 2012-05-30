function y = hankel_op_plus_history_and_bias( Un, m, i, N, s, x, mode )
% Strips the bias and history parameter off of the input and passes it to
% the linear operator that makes 

switch mode
    case 0 % return { input size, output size }
        y = { [ m, N + 1 + m*s ], [ m * i, size( Un, 2 ) ] };
    case 1 % apply operator to input
        y = block_hankel( x(:,1:end-1-m*s), 1, i, N ) * Un;
    case 2 % apply adjoint operator to input
        y = [ adjoint_hankel( x * Un', i, N ), zeros( m, 1 + m*s ) ];
end