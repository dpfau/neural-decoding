function [Oi yh] = build_proj( y, u, i, N, opts )
% From the input and output data, build the appropriate projections of the
% row space of the block Hankel matrix of outputs for doing subspace ID.

m = size(u,1);
if strcmpi( opts.proj, 'oblique' )
    assert( ~strcmp( opts.noise, 'none' ), 'Cannot use oblique projection with nuclear norm minimization' )
    Y = block_hankel( y, 1, 2*i, N );
    U = block_hankel( u, 1, 2*i, N );
    
    Yf = Y(i+1:end,:);
    Uf = U(i+1:end,:);
    
    Oi = oblique( Yf, Uf, [Y(1:i,:);U(1:i,:)] );
    yh = y;
elseif strncmpi( opts.proj, 'orth', 4 )
    Y = block_hankel( y, 1, i, N );
    U = block_hankel( u, 1, i, N );
    
    if strcmpi( opts.proj, 'orth_pinv' )
        Un = eye( N - i + 1 ) - pinv( U ) * U;
    elseif strcmpi( opts.proj, 'orth_svd' )
        [~,~,v] = svd(U);
        Un = v(:,m*i+1:end);
    end
    tic
    [Oi yh] = nucnrmmin( y(:,1:N), Y, Un, i, opts );
    toc
else
    error(['''' opts.proj ''' is not a recognized projection method.']);
end