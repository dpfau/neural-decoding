function [ll neg] = log_lik( x, b_1, b_inf, B_x, k, p, c, l )

if ~iscell( x )
    x = {x};
end

tiny = 1e-300;

ll = 0;
neg = 0;
for i = 1:length(x)
    b_t = b_1;
    for t = 1:length(x{i})
        [b_t like] = update( b_t, x{i}(t), k, p, c, l, b_inf, B_x );
        ll = ll + log(max(tiny,like));
        if like < tiny
            neg = neg + 1;
        end
    end
end