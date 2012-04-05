function [Q R] = fit_noise(e, A, C)

E0 = e*e'/size(e,2);
E1 = e(:,2:end)*e(:,1:end-1)'/(size(e,2)-1);

Q0 = randn(size(A));
Q0 = Q0*Q0'; % Initialize at a feasible point
opts = optimset('GradObj','on','Display','iter','LargeScale','on','Hessian','on','HessMult',@hess_mult);
S = fminunc(@(x) obj(x,E1,A,C,1), Q0, opts);
Q = S-A*S*A';
R = E0 - C*S*C';

function [f grad Hinfo] = obj(X,E,A,C,t)

foo = E-C*A*X*C';
bar = inv(X-A*X*A')';
f    = 0.5*norm(foo,'fro')^2 - t*log(det(bar));
grad = -A'*C'*foo*C - t*(bar-A'*bar*A);
Hinfo = struct('A',A,'C',C,'t',t,'bar',bar);

function Hv = hess_mult(Hinfo,v)

m = size(Hinfo.A,2);
n = size(Hinfo.C,2);
Hv = zeros(size(v));
A = Hinfo.A;
C = Hinfo.C;
t = Hinfo.t;
bar = Hinfo.bar;
for i = 1:size(v,2)
    sig = reshape(v(:,i),m,n);
    w = A'*(C'*C)*A*sig*(C'*C) + t*bar*(sig-A*sig*A')'*bar - t*A'*bar*(sig-A*sig*A')'*bar*A;
    Hv(:,i) = w(:);
end