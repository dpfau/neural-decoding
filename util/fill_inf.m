function y = fill_inf( x )
% Interpolate all NaN or Inf values

y = x;

for i = 1:size(y,1)
    idx = find( isnan( y(i,:) ) | isinf( y(i,:) ) );
    a = idx(1);
    b = 0;
    for j = 2:length( idx )
        if idx(j) ~= idx(j-1) + 1
        	b = idx(j-1);
            if a == 1
                y(i,1:b) = y(i,b+1);
            else
                y(i,a-1:b+1) = linspace(y(i,a-1),y(i,b+1),b-a+3);
            end
            a = idx(j);
        end
    end
    if idx(end) == length(y)
        y(i,a:end) = y(i,a-1);
    else
        b = idx(end);
        y(i,a-1:b+1) = linspace(y(i,a-1),y(i,b+1),b-a+3);
    end
end