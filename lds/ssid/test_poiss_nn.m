get_toy_data

opts.noise = 'poiss';
opts.proj = 'orth_svd';
opts.tol = '1e-3';
opts.maxIter = 100;
opts.instant = 0;
opts.vsig = 25;
opts.tfocs_path = '/Users/davidpfau/Documents/MATLAB/TFOCS';
opts.lag = 0;
moesp(x,u,5,1000,opts);