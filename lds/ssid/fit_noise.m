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
fprintf('Iter\tf(x)\n');
for t = 1:10
    opts = optimset('Algorithm','interior-point','GradObj','on','Display','on','Hessian','user-supplied','HessMult',@(X,l,v) hess_mult(X,A,C,2^-t,v));
    [S,fval] = fmincon(@(x) obj(x,E1,A,C,2^-t), S(:), [], [], symm, zeros(size(symm,1),1), [], [], [], opts);
    fprintf('%2.4d\t%2.4d\n',t,fval);
end
S = reshape(S,size(A));
Q = S-A*S*A';
R = E0 - C*S*C';

function [f grad Hinfo] = obj(X,E,A,C,t)

X = reshape(X,size(A));
foo = E-C*A*X*C';
bar = inv(X-A*X*A')';
f    = 0.5*norm(foo,'fro')^2 - t*log(det(bar));
grad = -A'*C'*foo*C - t*(bar-A'*bar*A);
Hinfo = struct('A',A,'C',C,'t',t,'bar',bar);

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