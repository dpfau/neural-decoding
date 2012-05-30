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
        tfocs_opts = tfocs_SCD;
        tfocs_opts.tol = 1e-4;
        tfocs_opts.maxIts = 1e3;
        tfocs_opts.printEvery = 0;
        yh_ = log( y(:,1:N) + 0.01 ); yh = Inf;
        while norm(yh-yh_) > 1e-6
            yh = yh_;
            yh_ = tfocs_SCD( @(varargin) poiss_loglik( y(:,1:N), s0(1)/l/N/opts.vsig^2, varargin{:} ), ...
                             @(varargin) hankel_op_plus_history_and_bias( Un, l, i, N, opts.lag, varargin{:} ), ...
                             @proj_spectral, ...
                             1e-5, ...
                             yh, ...
                             hankel_op( Un, l, i, N, yh, 1 ), ...
                             tfocs_opts );
        end
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
    otherwise
        error(['''' opts.noise ''' is not a recognized output noise.']);
end