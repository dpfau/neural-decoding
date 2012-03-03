function y = fill_inf( x )
% Interpolate all NaN or Inf values

y = x;
idx = find( isnan( x ) | isinf( x ) );
a = idx(1);
for i = 2:length( idx )
    if idx(i) ~= idx(i-1) + 1
        y(a-1:idx(i-1)+1) = y(a-1):(y(idx(i-1)+1)-y(a-1))/(idx(i-1)-a+2):y(idx(i-1)+1);
        a = idx(i);
    end
end