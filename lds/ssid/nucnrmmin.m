function [Oi yh] = nucnrmmin( y, Y, Un, i, opts )
% Depending on the options specified, returns the appropriate matrix for
% parameter reconstruction in subspace ID.  For no output noise, just
% return Y*Un.  For Gaussian output noise, perform nuclear norm
% minimization of Yh*Un, minimizing the distance between y and yh.  For
% Poisson output noise, do the same, but by ADMM.

[l,N] = size(y);
switch opts.noise
    case 'none'
        Oi = Y*Un;
        yh = y;
    case 'gauss'
        % For Gaussian noise, can do nuclear norm plus Gaussian
        % likelihood minimization directly
        addpath( opts.tfocs_path )
        s0 = svd(Y*Un);
        tfocs_opts = tfocs_SCD;
        tfocs_opts.tol = 1e-6;
        tfocs_opts.printEvery = 10;
        yh = tfocs_SCD( [], ...
            @(varargin) hankel_op( Un, l, i, N, varargin{:} ), ...
            @proj_spectral, ...
            2*s0(1)/l/N/opts.vsig^2, ...
            y(:,1:N), ...
            hankel_op( Un, l, i, N, y(:,1:N), 1 ), ...
            tfocs_opts );
        Oi = hankel_op( Un, l, i, N, yh, 1 );
    case 'poiss'
        addpath( opts.tfocs_path )
        s0 = svd(Y*Un);
        tfocs_opts = tfocs;
        tfocs_opts.tol = 1e-4;
        tfocs_opts.maxIts = 1e3;
        tfocs_opts.printEvery = 1;
        % smoothF = make_convex_conjugate( @(varargin) poiss_loglik( y(:,1:N), s0(1)/l/N/opts.vsig^2, varargin{:} ) );
        smoothF = @(varargin) poiss_loglik_conj( y(:,1:N), s0(1)/l/N/opts.vsig^2, varargin{:} );
        affineF = @(varargin) adjoint_hankel_op( Un, l, i, N, 0, varargin{:} );
        x0 = find_pos( affineF, size( y(:,1:N) ) );
        Zh = tfocs( smoothF, affineF, @proj_spectral, x0, tfocs_opts );
        [~,yh] = poiss_loglik_conj( y(:,1:N), s0(1)/l/N/opts.vsig^2, adjoint_hankel( Zh * Un', i, N ) );
        Oi = hankel_op( Un, l, i, N, yh, 1 );
    case 'poiss_bias'
        addpath( opts.tfocs_path )
        s0 = svd(Y*Un);
        tfocs_opts = tfocs;
        tfocs_opts.tol = 1e-4;
        tfocs_opts.maxIts = 1e3;
        tfocs_opts.printEvery = 1;
        f = make_convex_conjugate( @(x) poiss_loglik_bias( y(:,1:N), s0(1)/l/N/opts.vsig^2, 1, x ) );
        dim = hankel_op( Un, l, i, N, [], 0 );
        x0 = randn( dim{2} );
        Zh = tfocs( f, ...
            @(varargin) adjoint_hankel_op( Un, l, i, N, 1, varargin{:} ), ...
            @proj_spectral, ...
            x0, tfocs_opts );
        [~,yh] = poiss_loglik_conj( y(:,1:N), s0(1)/l/N/opts.vsig^2, adjoint_hankel( Zh * Un', i, N ) );
        Oi = hankel_op( Un, l, i, N, yh, 1 );
    case 'poiss_history'
        addpath( opts.tfocs_path )
        s0 = svd(Y*Un);
        tfocs_opts = tfocs;
        tfocs_opts.tol = 1e-4;
        tfocs_opts.maxIts = 1e3;
        tfocs_opts.printEvery = 1;
        f = make_convex_conjugate( @(x) poiss_loglik_bias_history( y(:,1:N), s0(1)/l/N/opts.vsig^2, 1, 1, opts.lag, x ) );
        Zh = tfocs( f, ...
            @(varargin) adjoint_hankel_op( Un, l, i, N, 1 + l*opts.lag, varargin{:} ), ...
            @proj_spectral, ...
            zeros( size( hankel_op( Un, l, i, N, y(:,1:N), 1 ) ) ), tfocs_opts );
        [~,yh] = poiss_loglik_conj( y(:,1:N), s0(1)/l/N/opts.vsig^2, adjoint_hankel( Zh * Un', i, N ) );
        Oi = hankel_op( Un, l, i, N, yh, 1 );
    case 'poiss_admm'
        % For Poisson noise, use ADMM to optimize jointly over nuclear
        % norm and Poisson likelihood
        addpath( opts.tfocs_path )
        s0 = svd(Y*Un);
        tfocs_opts = tfocs_SCD;
        tfocs_opts.tol = 1e-4;
        tfocs_opts.maxIts = 1e3;
        tfocs_opts.printEvery = 0;
        
        yh1 = y(:,1:N);
        b1 = log(mean(y,2) + 1e-6);
        zyh = zeros(size(yh1)); % auxilliary variable for ADMM
        zb  = zeros(size(b1));
        r_norm = Inf; % residual
        lambda = 2*s0(1)/l/N/opts.vsig^2; % tradeoff between nuclear norm and output log likelihood
        
        fprintf('%3s\t%10s\t%10s\t%10s\n', 'iter', ...
            'r norm', 's norm', 'objective');
        iter = 0;
        while r_norm > 1e-4 % ADMM loop
            iter = iter + 1;
            % Nuclear norm minimization
            yh = tfocs_SCD( [], ...
                @(varargin) hankel_op( Un, l, i, N, varargin{:} ), ...
                @proj_spectral, ...
                opts.rho, ...
                yh1 - zyh, ...
                hankel_op( Un, l, i, N, yh1, 1 ), ...
                tfocs_opts );
            b = b1;
            
            % Minimize augmented negative log likelihood
            opts1 = struct( 'GradObj', 'on', ...
                'LargeScale', 'on', ...
                'Hessian', 'on', ...
                'HessMult', @hessmult, ...
                'Display', 'off' );
            xold = [yh1, b1];
            x1 = fminunc( @(x) aug_ll( x, [yh,b], lambda, opts.rho, y(:,1:N), [zyh,zb] ), [yh,b], opts1 );
            yh1 = x1(:,1:end-1);
            b1  = x1(:,end);
            
            % Dual variable update
            zyh = zyh + yh - yh1;
            zb  = zb  + b - b1;
            r_norm = norm( yh - yh1, 'fro' ) + norm( b - b1, 'fro' );
            s_norm = norm( -opts.rho*( [yh1, b1] - xold ), 'fro' );
            obj = objective( yh, yh1, b1, y(:,1:N), lambda, i, Un );
            fprintf('%3d\t%10.4f\t%10.4f\t%10.2f\n', iter, r_norm, s_norm, obj );
        end
        Oi = hankel_op( Un, l, i, N, yh, 1 );
    case {'gauss_lhv','poiss_lhv'} % The ADMM method of Liu, Hansson and Vandenberghe, 2012
        % set default values for stopping criteria
        if isfield(opts, 'rho')
            rho = opts.rho;
        else
            rho = 1.3;
        end
        if isfield(opts, 'eps_abs')
            eps_abs = opts.eps_abs;
        else
            eps_abs = 1e-6;
        end
        if isfield(opts, 'eps_rel')
            eps_rel = opts.eps_rel;
        else
            eps_rel = 1e-3;
        end
        A     = @(x) block_hankel( x, 1, i, N ) * Un; % No instrumental variables or weighting matrices here, to add later.
        A_adj = @(x) adjoint_hankel( x * Un', i, N );
        
        % s0 = svd(Y*Un);
        p = l*i;
        q = size(Un,2);
        X = zeros(p,q);
        Z = ones(p,q);
        x = log(y + 1e-3);
        
        r_p = Inf; r_d = Inf;
        e_p = 0; e_d = 0;
        iter = 0;
        fprintf('Iter:\t objective:\t aug lgrn:\t r_p:\t\t e_p:\t\t r_d:\t\t e_d:\n')
        while norm( r_p, 'fro' ) > e_p || norm( r_d ) > e_d
            
            % Naive implementation that does not take advantage of speedup described in LHV2012
            if strcmp(opts.noise,'poiss_lhv')
                x_ = conj_grad( @(w) opts.lambda * exp( x ) .* w + rho * A_adj( A( w ) ), ...
                                A_adj( rho * X - Z ) + opts.lambda * exp( x ) .* ( -y + exp( x ) ) );
            else
                x_ = conj_grad( @(w) opts.lambda * reshape( opts.H * w(:), size( w ) ) + rho * A_adj( A( w ) ), ...
                                A_adj( rho * X - Z ) + opts.lambda * reshape( opts.H * opts.a, size( x ) ) );
            end
            Ax_ = A( x_ );
            
            [u,s,v] = svd( Ax_ + Z/rho );
            X_ = u*max( s - eye(p,q)/rho, 0 )*v';
            
            Z_ = Z + rho * ( Ax_ - X_ );
            
            % compute residuals and thresholds
            r_p = Ax_ - X_;
            r_d = rho * A_adj( X - X_ );
            e_p = sqrt(p*q) * eps_abs + eps_rel * max( norm( Ax_, 'fro' ), norm( X_, 'fro' ) );
            e_d = sqrt(l*N) * eps_abs + eps_rel * norm( A_adj( Z ), 'fro' );
            
            % update
            X = X_;
            Z = Z_;
            x = x_;
            if strcmp(opts.noise,'poiss_lhv')
                f = opts.lambda * sum( sum( -y .* x + exp( x ) ) );
            else
                f = opts.lambda/2 * ( x(:) - opts.a )' * opts.H * ( x(:) - opts.a );
            end
            obj = sum( max( svd( Ax_ ) - 1 / rho, 0 ) ) + f;
            aug_lgrn = sum( max( diag( s ) - 1 / rho, 0 ) ) + f + trace( Z' * ( Ax_ - X ) ) + rho/2 * norm( Ax_ - X, 'fro' );
            
            % print
            iter = iter + 1;
            fprintf('%i\t %1.2d\t %1.2d\t %1.2d\t %1.2d\t %1.2d\t %1.2d\n', iter, obj, aug_lgrn, norm( r_p, 'fro' ), e_p, norm( r_d ), e_d);
        end
        yh = x;
        Oi = A( yh );
    otherwise
        error(['''' opts.noise ''' is not a recognized output noise.']);
end