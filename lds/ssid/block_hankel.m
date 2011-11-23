function y = block_hankel( x, i, s, n )

k = size(x,1);
assert( size(x,2) >= n );
y = zeros(s*k, n-s+1);

for t = 0:(s-1)
    y(t*k+(1:k),:) = x(:,(i+t):(i+t+n-s));
end