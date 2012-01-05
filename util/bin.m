function [s, c, d] = bin( spikes, dt, cvr )
% spikes - cell array of spike times for each unit
% dt - bin width
% cvr - optional cell array of covariates, first column is times, second 
%       column values
% s - binned spike times
% c - binned covariates
% d - binned derivatives of covariates

c = [];
T0 = min( cellfun( @(x) min(x(:,1)), spikes ) );
T1 = max( cellfun( @(x) max(x(:,1)), spikes ) );
if nargin == 3
    T0 = min( [ T0, cellfun( @(x) min( x(:,1) ), cvr ) ] );
    T1 = max( [ T1, cellfun( @(x) max( x(:,1) ), cvr ) ] );
end
edges = T0:dt:T1+dt;
s = zeros( length( edges ), size( spikes, 1 ) );

for j = 1:length( spikes )
    s(:,j) = histc( spikes{j}(:,1), edges );
end

if nargin == 3
    [c,d] = fast_avg( edges, cvr );
end