function [ll grad Hinfo] = log_lik( data, x, params )

coord = interp_coord( x, length(params.template) );
ll = sum( -( interp( coord, params.template, 0 ) - data ).^2/(2*params.sig_temp^2) ) + ...
    sum( -( (x(2:end)-x(1:end-1)) - params.dt ).^2/(2*params.sig_dt^2) ) + ...
    -( x(1) - params.t0 )^2/(2*params.sig_dt^2) + ...
    - length(data)*log(params.sig_temp) - length(data)*log(params.sig_dt) - length(data)*log(2*pi);

grad = zeros(size(x));
grad(1) = (params.t0 - x(1))/params.sig_dt^2;
grad(1:end-1) = grad(1:end-1) - ( x(1:end-1) + params.dt - x(2:end) )/params.sig_dt^2;
grad(2:end)   = grad(2:end)   - ( x(2:end) - params.dt - x(1:end-1) )/params.sig_dt^2;
grad = grad + ( data - interp( coord, params.template, 0 ) ).*interp( coord, params.template, 1 )/params.sig_temp^2;

Hinfo = struct('diag',-2/params.sig_dt^2*ones(size(grad)),'off_diag',ones(1,size(grad,2)-1)/params.sig_dt^2);
Hinfo.diag(end) = -1/params.sig_dt^2;
Hinfo.diag = Hinfo.diag + ( -(interp( coord, params.template, 1 ).^2) + ...
    ( data - interp( coord, params.template, 0 ) ).*interp( coord, params.template, 2 ) )/params.sig_temp^2;

ll = -ll; 
grad = -grad; % Hack because I want to maximize, not minimize.
Hinfo.diag = -Hinfo.diag; 
Hinfo.off_diag = -Hinfo.off_diag;