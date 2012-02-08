function [init cross] = make_init( x, thresh, up, a )
% Given a sequence x, makes a reasonable guess for the phase sequence
% of a latent-phase model just by counting the number of times a threshold
% is crossed from above or below
%
% x - the sequence
% thresh - the threshold
% up - whether the threshold crossing is from above or below
% a - rate constant for low-pass filtering
% David Pfau, 2012

filt_x = filter(a,[1,a-1],x);
if up
    cross = find( filt_x(1:end-1) > thresh & filt_x(2:end) <= thresh );
else
    cross = find( filt_x(1:end-1) <= thresh & filt_x(2:end) > thresh );
end
T = mean(diff(cross));
init = zeros(size(x));
init(1:cross(1)-1) = ((-cross(1)+2):0)*2*pi/T;
for i = 1:length(cross)-1
    init(cross(i):cross(i+1)-1) = 2*pi*(i-1)+(1:cross(i+1)-cross(i))*2*pi/(cross(i+1)-cross(i));
end
init(cross(end):end) = 2*pi*(length(cross)-1)+(1:length(init)-cross(end)+1)*2*pi/T;