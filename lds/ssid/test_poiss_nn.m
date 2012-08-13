%get_toy_data
x = dlmread('x.txt');
u = dlmread('u.txt');

N = 200;
opts.noise = 'poiss_lhv';

opts.H = sparse(eye(size(x,1)*N));
opts.a = x(:,1:N) + 1;
opts.g = ones( size( opts.a ) );
opts.f0 = 0;

opts.proj = 'orth_svd';
opts.tol = 1e-3;
opts.maxIter = 100;
opts.instant = 0;
opts.lambda = 2;
opts.rho = 1;
opts.tfocs_path = '/Users/davidpfau/Documents/MATLAB/TFOCS';
opts.lag = 0;
moesp(x,u,5,N,opts);