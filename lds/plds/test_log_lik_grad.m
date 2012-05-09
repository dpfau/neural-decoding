function test_log_lik_grad( data, map, params, eps )

if nargin < 4
    eps = 1e-8;
end
[ll,grad,~] = log_lik( data, map, params );
for i = 1:numel(map)
    map(i) = map(i) + eps;
    ll_ = log_lik( data, map, params );
    map(i) = map(i) - eps;
    fprintf('Exact grad: %2.4d\t Approx grad: %2.4d\t Diff: %2.4d\n',grad(i),(ll_-ll)/eps,grad(i)-(ll_-ll)/eps);
end