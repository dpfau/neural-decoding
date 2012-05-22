function [z a b c w] = sgplvm(y,d)
% Implements Scaled Gaussian Process Latent Variable Model to project data 
% y into d-dimensional subspace using RBF kernel with parameters a,b,c:
%   k(z,z') = a*exp( -c/2*||z-z'||^2 ) + delta(z,z')/b

D = size(y,1);
N = size(y,2);
mu = mean(y,2);
y = y-mu(:,ones(1,N)); % center data
w = ones(1,D); % scaling vector
a = 1;
b = 1;
c = 1;
[~,~,z] = svd(y);
z = z(:,1:d)';
test_grad(y,struct('a',a,'b',b,'c',c,'z',z,'w',w));

function [fy grad] = sgplvm_obj(y,params)
% Computes the objective as well as gradient wrt z,a,b,c,w of the objective
% for the scaled GPLVM

z = params.z; a = params.a; b = params.b; c = params.c; w = params.w;
assert( size(z,2) == size(y,2), 'Latent state has wrong dimension' )
D = size(y,1);
N = size(y,2);
[K d e] = kernel(z,a,b,c);
wy = diag(w)*y;
Ki = K^-1; % This is the biggest impediment to scaling
fy = D*sum(diag(chol(K))) ...
     + 1/2*sum(diag(wy*Ki*wy')) ...
     + 1/2*z(:)'*z(:) ...
     + log(a) + log(b) + log(c) ...
     - N*sum(log(w));
dK = (Ki*wy')*(wy*Ki) - D*Ki;
dz = -c*(tprod(K,[2 3],z,[1 2])-tprod(K,[2 3],z,[1 3]));
grad = struct( 'z', tprod(dK,[2 -1],dz,[1 2 -1]) + z, ...
               'a', trace(dK*e) + 1/a, ...
               'b', trace(-dK/b^2) + 1/b, ...
               'c', trace(-1/2*dK*(d.*K)) + 1/c, ...
               'w', w.*diag(y*Ki*y')'-N*sum(1./w) );
           
function test_grad(y,params)

[f,grad] = sgplvm_obj(y,params);
for var = fieldnames(params)'
    fprintf('%s:\n', char(var))
    for i = 1:numel(params.(char(var)))
        params.(char(var))(i) = params.(char(var))(i) + 1e-5;
        f_ = sgplvm_obj(y,params);
        params.(char(var))(i) = params.(char(var))(i) - 1e-5;
        fprintf('Numeric: %d, Analytic: %d\n',(f_-f)/1e-5,grad.(char(var))(i));
    end
end