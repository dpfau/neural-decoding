function y = adjoint_hankel( Y, i, N )
% From the linear operation that turns y into a block-hankel matrix with i
% block-rows and uses N columns of y, perform the adjoint of that operation.

m = size(Y,1);
y = zeros( m * N / i, 1 );
for t = 1:size(Y,2)
    y( (t-1)*m/i + (1:m) ) = y( (t-1)*m/i + (1:m) ) + Y(:,t);
end
y = reshape( y, m / i, N );