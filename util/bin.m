function [s, c, d] = bin( spikes, dt, cvr )
% Given spike times and covariates with time stamps, aligns and bins the
% data such that it can be easily passed to some decoding algorithm such as
% a Kalman filter.  While the analyze framework probably already has
% scripts to do this, I'm still learning my way around analyze and will use
% this for the time being.
%
% spikes - cell array of spike times for each unit
% dt - bin width
% cvr - optional cell array of covariates, first column is times, second 
%       column values
% s - binned spike times
% c - binned covariates
% d - binned derivatives of covariates
%
% David Pfau, 2011-2012

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
    if exist('fast_avg','file') == 3
       [c,d] = fast_avg( edges, cvr );
    elseif exist('fast_avg.c','file') == 2
       mex fast_avg.c
       [c,d] = fast_avg( edges, cvr );
    else
        error( 'Missing mex file for fast_avg!' );
    end
end