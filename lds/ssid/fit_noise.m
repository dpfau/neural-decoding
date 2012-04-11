function [Q R] = fit_noise(e, A, C, k)
% Estimate the noise covariances for the linear dynamical system:
% x(t+1) = A*x(t) + B*u(t) + v(t)
% y(t)   = C*x(t) + D*u(t) + w(t)
% v(t) ~ N(0,Q)
% w(t) ~ N(0,R)
% By looking at the covariance between residuals at different time steps,
% where the residuals are the difference between the observed y(t) and the
% estimated y(t) assuming zero noise.
%
% Input:
%   e - the difference between the observed y and the y reconstructed from
%       a noiseless linear dynamical system
%   C - the estimated latent-state-to-output matrix
%   A - the estimated latent state evolution matrix
%   k - the maximum time lag for which we use the covariance.  If dim(x) <
%       dim(y) then asymptotically we'd never need k > 1, but in practice it
%       helps
%
% Output:
%   Q - Latent state noise, which introduces covariance between residuals
%       at different time steps
%   R - Output noise, which is uncorrelated from step to step
%
% David Pfau, 2012

E0 = e*e'/size(e,2);
E = cell(k,1);
for i = 1:k
    E{i} = e(:,1+i:end)*e(:,1:end-i)'/(size(e,2)-i);
end

Q0 = randn(size(A));
symm = zeros(size(Q0,1)*(size(Q0,1)-1)/2,numel(Q0));
t=0;
for i = 1:size(Q0,1)
    for j = i+1:size(Q0,1)
        t = t+1;
        symm(t,sub2ind(size(Q0),i,j)) = 1;
        symm(t,sub2ind(size(Q0),j,i)) = -1;
    end
end
S = eye(size(A,1))  - Q0 + A*Q0*A';
T = eye(size(E0,1)) - E0 + C*Q0*C';
st = numel(S)+numel(T);
t0 = main_obj(Q0,A,C,E);
Sig = constrained_newton(@(x) obj_phase_1(x,E0,E,A,C,t0), ...
                        [Q0(:);S(:);T(:)], ...
                        [symm, zeros(size(symm,1),st); zeros(st,numel(Q0)), eye(st)], ...
                        zeros(size(symm,1)+st,1), ...
                        1e-6);
                    
fprintf('Iter\tf(x)\t\tmax(imag(eig))\tmin(real(eig))\n');
fprintf('%2.4d\t%2.4d\t%2.4d\t%2.4d\n',0,obj(S(:),E0,E,A,C,1),max(imag(eig(reshape(S,size(A))))),min(real(eig(reshape(S,size(A))))));
for t = 1:50
    [Sig,fval] = constrained_newton(@(x) obj(x,E0,E,A,C,t0*2^-t), Sig(1:numel(Q0)), symm, zeros(size(symm,1),1), 1e-6);
    fprintf('%2.4d\t%2.4d\t%2.4d\t%2.4d\n',t,fval,max(imag(eig(reshape(S,size(A))))),min(real(eig(reshape(S,size(A))))));
end
Sig = reshape(Sig,size(A));
Q = Sig-A*Sig*A';
R = E0 -C*Sig*C';

function [f grad hess] = obj(X,E0,E,A,C,t)

X = reshape(X,size(A));
[main_f main_grad main_hess] = main_obj(X,A,C,E);
[bar1_f bar1_grad bar1_hess] = barrier(X, @(x) x-A*x*A', @(x) -x+A'*x*A, 0);
[bar2_f bar2_grad bar2_hess] = barrier(X, @(x)  -C*x*C', @(x)    C'*x*C, E0);
f    = main_f    + t*bar1_f    + t*bar2_f;
grad = main_grad + t*bar1_grad + t*bar2_grad;
hess = main_hess + t*bar1_hess + t*bar2_hess;

function [f grad hess] = obj_phase_1(x,E0,E,A,C,t)

m = size(A,1);
n = size(C,1);
X = reshape(x(1:m^2),m,m);
S = reshape(x(m^2+1:2*m^2),m,m);
T = reshape(x(2*m^2+1:end),n,n);
[main_f main_grad main_hess] = main_obj(X,A,C,E);
[bar1_f bar1_grad bar1_hess] = barrier(X, @(x) x-A*x*A', @(x) -x+A'*x*A, S);
[bar2_f bar2_grad bar2_hess] = barrier(X, @(x)  -C*x*C', @(x)    C'*x*C, E0+T);
f       = main_f    + t*bar1_f    + t*bar2_f;
grad_X  = main_grad + t*bar1_grad + t*bar2_grad;
hess_XX = main_hess + t*bar1_hess + t*bar2_hess;

[~,grad_S,hess_SS] = barrier(S, @(x) x, @(x) -x, X-A*X*A');
[~,grad_T,hess_TT] = barrier(T, @(x) x, @(x) -x, E0-C*X*C');
hess_SX = hess_mult_to_hess(@(x) inv(S+X-A*X*A')'*(x-A*x*A')'*inv(S+X-A*X*A')',size(A));
hess_TX = hess_mult_to_hess(@(x) inv(T+E0-C*X*C')'*(-C*x*C')'*inv(T+E0-C*X*C')',size(C));

grad = [grad_X(:); t*grad_S(:); t*grad_T(:)];
hess = [hess_XX,   t*hess_SX',     t*hess_TX'; ...
        t*hess_SX, t*hess_SS,      zeros(m^2,n^2); ...
        t*hess_TX, zeros(n^2,m^2), t*hess_TT];

function [f grad hess] = main_obj(X,A,C,E)

k = length(E);
f = 0;
grad = zeros(size(X));
hess = zeros(size(X,1)^2,size(X,2)^2);
for i = 1:k
    foo = E{i}-C*A^i*X*C';
    f = f + 0.5*norm(foo,'fro')^2;
    grad = grad - (A^i)'*C'*foo*C;
    hess = hess + hess_mult_to_hess(@(x) (A^i)'*(C'*C)*(A^i)*x*(C'*C), size(X));
end

function [f grad hess] = barrier(X,g,h,c)

foo = g(X)+c;
bar = inv(foo)';
f = -log(det(foo));
grad = h(bar);
hess = hess_mult_to_hess(@(x) -h(bar*g(x)'*bar), size(X));

function hess = hess_mult_to_hess(hm,n)

hess = zeros(n(1)^2,n(2)^2);
for i = 1:n(2)^2
    foo = zeros(n(2),n(2));
    foo(i) = 1;
    bar = hm(foo);
    hess(:,i) = bar(:);
end

function [dx dx_] = test_grad(f,x)

[fx,dx,~] = f(x);
dx_ = zeros(size(dx));
for i = 1:numel(x);
    x(i) = x(i) + 1e-8;
    dx_(i) = (f(x)-fx)*1e8;
    x(i) = x(i) - 1e-8;
end

function [H H_] = test_hess(f,x)

[~,grad,H] = f(x);
H_ = zeros(size(H));
for i = 1:numel(x)
    x(i) = x(i) + 1e-8;
    [~,grad_,~] = f(x);
    H_(:,i) = (grad_(:)-grad(:))*1e8;
    x(i) = x(i) - 1e-8;
end