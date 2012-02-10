function params = em( data, eps, thresh, up, N )
% Expectation-Maximiation for latent-phase model
% David Pfau 2012

if nargin < 3
    figure(1); plot( data );
    R = input('Enter threshold\n','s');
    thresh = str2double(R);
    R = input('Enter direction of threshold crossing\n','s');
    up = str2double(R);
    R = input('Enter template size\n','s');
    N = str2double(R);
end

map = make_init( data, thresh, up, .1 ); % initial estimate for phase path
map = map(1:end-1);
params = m_step( diff(data), N, map );
ll = Inf;
ll_ = log_lik( diff(data), map, params );
i = 0;
while i < 10 || ll - ll_ > eps
    i = i+1;
    ll = ll_;
    [map,prec,ll_] = e_step( diff(data), params, map ); % initialize with path from previous step
    params = m_step( diff(data), N, map );
    fprintf('Iter %i - Log lik: %d\n',i,ll_);
end