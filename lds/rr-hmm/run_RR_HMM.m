addpath '/Users/davidpfau/Documents/Pesaran Group/data/111221/006'
addpath '/Users/davidpfau/Documents/Pesaran Group/data/111221/005'

load rec006.Body.Marker.mat
train = zeros(3,size(Marker{7},2)/10);
for i = 1:size(train,2) % decimate training data
    train(:,i) = mean( Marker{7}(2:4,(i-1)*10+(1:10)), 2 );
end

load rec005.Body.Marker.mat
test = zeros(3,size(Marker{7},2)/10);
for i = 1:size(test,2) % decimate testing data
    test(:,i) = mean( Marker{7}(2:4,(i-1)*10+(1:10)), 2 );
end
l = size(train,1);

n = 20; % number of past and future observations used for training
k = 20; % number of dimensions we project past and future observations onto
nkern = 200;
ncv = 50; % number of points used for bandwidth cross-validation

idx_kern = ceil( rand(nkern+ncv,1)*(size(train,2)-2*n) ); % random kernel centers, including some used for bandwidth cross-validation
train_stacked = block_hankel( train, 1, 2*n+1, size(train,2) );
kern_past    = train_stacked( 1:l*n, idx_kern(1:nkern) );
kern_present = train_stacked( l*n+(1:l), idx_kern(1:nkern) );
kern_future  = train_stacked( (n+1)*l+1:end, idx_kern(1:nkern) );

disp('Estimating kernel parameters...')
[u,s,~] = svd( kern_past - mean(kern_past,2)*ones(1,nkern) );
proj_past = s(1:k,1:k)^-1*u(:,1:k)';

mean_present = mean(kern_present,2);
centered_present = kern_present - mean_present*ones(1,nkern);
Q = chol(centered_present*centered_present'); % Used for generating from the predictive distribution
[u,s,~] = svd( centered_present );
proj_present = s(1:l,1:l)^-1*u';

disp('Finding kernel bandwidth by cross-validation...')
kern_cv = train(:,n+idx_kern(nkern+1:end)); % points used for cross-validation
proj_kern = proj_present*kern_present; % project kernel centers into subspace w/isotropic kernels
proj_cv = proj_present*kern_cv; % project cross-validation data into subspace with isotropic kernels
cv_norm = zeros(nkern,ncv);
for i = 1:nkern
    for j = 1:ncv
        cv_norm(i,j) = norm( proj_kern(:,i) - proj_cv(:,j) )^2;
    end
end

cv_lik = zeros(100,1);
for prec = 1:100
    cv_lik(prec) = prod( prec^l/(nkern*(pi*2)^l/2)*sum( exp( -prec^2*cv_norm/2 ) ) );
end
[~,prec] = max(cv_lik); % set precision to maximum held-out likelihood
Q = Q/prec;

[u,s,~] = svd( kern_future - mean(kern_future,2)*ones(1,200) );
proj_future = s(1:k,1:k)^-1*u(:,1:k)';

disp('Estimating RR-HMM parameters...')
k = @(x) exp( -column_squared_norm(x) );
[b_1, b_inf, B_x] = est_RR_HMM( {train}, 10, k, ...
    kern_past, kern_present, kern_future, ...
    proj_past, proj_present, proj_future, 1/prec );

disp('Calculating training log likelihood...')
[train_log_like train_neg] = log_like( train, b_1, b_inf, B_x, k, proj_present, kern_present, l );
disp('Calculating testing log likelihood...')
[test_log_lik   test_neg]  = log_like( test, b_1, b_inf, B_x, k, proj_present, kern_present, l );


disp('Generating data from model...')
b_t = b_1;
gen_from_model = zeros(l,1000);
for i = 1:1000
    gen_from_model(:,i) = generate( b_t, Q, c, b_inf, B_x );
    b_t = update( b_t, gen_from_model(:,i), k, proj_present, kern_present, 1/prec, b_inf, B_x );
end
