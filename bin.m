function [s, c] = bin( spikes, dt, cvr )
% spikes - cell array of spike times for each unit
% dt - bin width
% cvr - optional cell array of covariates, first column is times, second 
%       column values
% s - binned spike times
% c - binned covariates

c = [];
T0 = min( cellfun( @min, spikes ) );
T1 = max( cellfun( @max, spikes ) );
if nargin == 3
    T0 = min( [ T0, cellfun( @(x) min( x(:,1) ), cvr ) ] );
    T1 = max( [ T1, cellfun( @(x) max( x(:,1) ), cvr ) ] );
end
edges = T0:dt:T1+dt;
s = zeros( length( edges ), length( spikes ) );

for j = 1:length( spikes )
    s(:,j) = histc( spikes{j}, edges );
end

if nargin == 3
    c = zeros( length( edges ), length( cvr ) );
    for j = 1:length( cvr )
        [~,bin] = histc( cvr{j}(:,1), edges );
        for i = 1:max(bin)
            c(i,j) = mean( cvr{j}( bin==i, 2 ) );
        end
    end
end