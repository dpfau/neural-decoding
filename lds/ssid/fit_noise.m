function [Q R S] = fit_noise(e, A, C)

E0 = e*e'/size(e,2);
E1 = e(:,2:end)*e(:,1:end-1)'/(size(e,2)-1);

S = randn(size(A));
S = S*S'; % Initialize at a feasible point
symm = zeros(size(S,1)*(size(S,1)-1)/2,numel(S));
t = 0;
for i = 1:size(S,1)
    for j = i+1:size(S,1)
        t = t+1;
        symm(t,sub2ind(size(S),i,j)) = 1;
        symm(t,sub2ind(size(S),j,i)) = -1;
    end
end
fprintf('Iter\tf(x)\tmax(imag(eig))\tmin(real(eig))\n');
fprintf('%2.4d\t%2.4d\t%2.4d\t%2.4d\n',0,obj(S(:),E1,A,C,1),max(imag(eig(reshape(S,size(A))))),min(real(eig(reshape(S,size(A))))));
for t = 1:100
    [S,fval] = constrained_newton(@(x) obj(x,E1,A,C,1000), S(:), symm, zeros(size(symm,1),1), 1e-8);
    fprintf('%2.4d\t%2.4d\t%2.4d\t%2.4d\n',t,fval,max(imag(eig(reshape(S,size(A))))),min(real(eig(reshape(S,size(A))))));
end
S = reshape(S,size(A));
Q = S-A*S*A';
R = E0 - C*S*C';

function [f grad hess] = obj(X,E,A,C,t)

X = reshape(X,size(A));
foo = E-C*A*X*C';
bar = inv(X-A*X*A')';
f    = 0.5*norm(foo,'fro')^2 - t*log(det(bar));
grad = -A'*C'*foo*C - t*(bar-A'*bar*A);
hess = zeros(numel(X));
for i = 1:numel(X)
    foo = zeros(size(hess,2),1); 
    foo(i) = 1; 
    hess(:,i) = hess_mult(X,A,C,t,foo);
end

function [x fx] = constrained_newton(f,x0,A,b,eps)

assert( size(A,1) == size(b,1), 'A and b must have same number of rows' );
assert( size(b,2) == 1, 'b must be vector' );
x = x0;
fx_ = Inf;
[fx,grad,hess] = f(x);
while abs(fx - fx_) > eps
    foo = [hess, A'; A, zeros(numel(b))]\[-grad(:); b];
    dx = foo(1:length(x));
    a = 1;
    while f(x + a*dx) > fx + eps
        a = a/2;
    end
    x = x + a*dx;
    fx_ = fx;
    [fx,grad,hess] = f(x);
end
    
function Hv = hess_mult(X,A,C,t,v)

X = reshape(X,size(A));
m = size(A,2);
n = size(C,2);
Hv = zeros(size(v));
bar = inv(X-A*X*A')';
for i = 1:size(v,2)
    sig = reshape(v(:,i),m,n);
    w = A'*(C'*C)*A*sig*(C'*C) + t*bar*(sig-A*sig*A')'*bar - t*A'*bar*(sig-A*sig*A')'*bar*A;
    Hv(:,i) = w(:);
end