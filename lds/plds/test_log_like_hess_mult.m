function test_log_like_hess_mult( data, map, params )

[~,grad,Hinfo] = log_lik( data, map, params );
for i = 1:numel(map)
    v = zeros(size(map));
    v(i) = 1;
    map(i) = map(i) + 1e-8;
    [~,grad_,~] = log_lik( data, map, params );
    map(i) = map(i) - 1e-8;
    e_hess = hess_mult(Hinfo,v);
    a_hess = (grad_-grad)/1e-8;
    fprintf('Exact hess: %d\t Approx hess: %d\t Diff: %d\n',norm(e_hess),norm(a_hess),norm(e_hess-a_hess));
end