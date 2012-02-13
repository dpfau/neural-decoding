function [x y] = gen( params, len )
% Generate data from the model

N = length( params.template );
x = zeros(len,1);
y = zeros(len,1);
phase = params.t0 + params.sig_dt*randn;
for i = 1:len
    x(i) = interp( interp_coord( phase, N ), params.template, 0 ) + params.sig_temp*randn;
    y(i) = phase;
    phase = phase + params.dt + params.sig_dt*randn;
end