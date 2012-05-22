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
[z,~] = eigs(y*y',d);

function [fy grad] = sgplvm_obj(y,z,a,b,c,w)
% Computes the objective as well as gradient wrt z,a,b,c,w of the objective
% for the scaled GPLVM

D = size(y,1);
N = size(y,2);
K = zeros(N);
for i = 1:N
    for j = 1:N
        K(i,j) = kernel(z(:,i),z(:,j),a,b,c,i==j);
    end
end
wy = diag(w)*y;
fy = D/2*sum(diag(chol(K))) ...
     + 1/2*sum(diag(wy'*(K\wy))) ...
     + 1/2*z(:)'*z(:) ...
     + log(a) + log(b) + log(c) ...
     - N*sum(log(w));
grad = struct('z',[],'a',[],'b',[],'c',[],'w',[]);

function k = kernel(z1,z2,a,b,c,i)

k = a*exp( -c/2*(z1-z2)'*(z1-z2) ) + i/b;