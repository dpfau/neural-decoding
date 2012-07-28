function f = make_linop( A )
    % Makes the trivial linear operator that is just matrix multiplication
    function y = linop( x, t )
        if t == 0
            y = { [size(A,2) 1], [size(A,1) 1] };
        elseif t == 1
            y = A*x;
        elseif t == 2
            y = A'*x;
        else
            error('Not a recognized mode for TFOCS linear operator')
        end
    end
    f = @linop;
end