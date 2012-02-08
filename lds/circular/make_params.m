function params = make_params( x, init )
t0 = init(1);
dt = mean(diff(init));
sig_dt = std(diff(init));

params = struct('t0',t0,'dt',dt,'sig_dt',sig_dt,'sig_temp',sig_temp,'template',template);