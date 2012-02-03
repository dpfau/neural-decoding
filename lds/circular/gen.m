function x = gen( params, len )
% Generate data from the model

x = zeros(len,1);
phase = params.t0 + params.sig_dt*randn;
for i = 1:len
    x(i) = interp( phase, params.template ) + params.sig_temp*randn;
    phase = phase + params.dt + params.sig_dt*randn;
end

function y = interp( x, template )

N = length(template);
phase = mod( x, 2*pi )*N/(2*pi); % map all points on the path into [0,length(template))
coord = ones(N,1)*phase - (1:N)'*ones(1,size(phase,2));
coord( coord > N/2 )  = coord( coord > N/2 )  - N; % shift difference between phase and template points from (-2*pi,2*pi) to (-pi,pi)
coord( coord < -N/2 ) = coord( coord < -N/2 ) + N;
y = template'*sinc(coord) ; % Whittaker-Shannon interpolation