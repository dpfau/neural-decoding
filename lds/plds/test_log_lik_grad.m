function test_log_lik_grad( data, map, params )

[ll,grad,~] = log_lik( data, map, params );
for i = 1:numel(map)
    map(i) = map(i) + 1e-8;
    ll_ = log_lik( data, map, params );
    map(i) = map(i) - 1e-8;
    fprintf('Exact grad: %d\t Approx grad: %d\t Diff: %d\n',grad(i),(ll_-ll)/1e-8,grad(i)-(ll_-ll)/1e-8);
end