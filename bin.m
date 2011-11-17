function [s, c, d] = bin( spikes, dt, cvr )
% spikes - cell array of spike times for each unit
% dt - bin width
% cvr - optional cell array of covariates, first column is times, second 
%       column values
% s - binned spike times
% c - binned covariates
% d - binned derivatives of covariates

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
    [c,d] = fast_avg( edges, cvr );
end