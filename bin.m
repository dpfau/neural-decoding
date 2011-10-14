function s = bin( spikes, dt )

T = max( cellfun( @max, spikes ) );
s = zeros( ceil( T/dt ), length( spikes ) );

for j = 1:length( spikes )
    s(:,j) = hist( spikes{j}, size(s,1) );
end