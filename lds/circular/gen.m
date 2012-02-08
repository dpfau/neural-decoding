function [x y] = gen( params, len )
% Generate data from the model

x = zeros(len,1);
y = zeros(len,1);
phase = params.t0 + params.sig_dt*randn;
for i = 1:len
    x(i) = interp( phase, params.template ) + params.sig_temp*randn;
    y(i) = phase;
    phase = phase + params.dt + params.sig_dt*randn;
end

function y = interp( x, template )

N = length(template);
phase = mod( x, 2*pi )*N/(2*pi); % map all points on the path into [0,length(template))
coord = ones(N,1)*x*N/(2*pi) - (1:N)'*ones(1,size(phase,2));
y = template'*(sin(pi*coord).*cot(pi*coord/N)/N) ; % Whittaker-Shannon interpolation