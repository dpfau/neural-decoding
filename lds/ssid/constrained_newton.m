function [x fx] = constrained_newton(f,x0,A,b,eps,mode,test)
% Primal-dual method for equality-constrained Newton's method with
% infeasible start.  Test is a function that is less than 0 if some
% constraint is violated, which is something of a hack, but turns out to be
% necessary in some cases.

warning('off','MATLAB:nearlySingularMatrix')
if nargin == 5 || ~strcmp(mode,'verbose')
    verbose = 0;
else
    verbose = 1;
end
if nargin < 7
    test = @(x) 1;
end
assert( size(A,1) == size(b,1), 'A and b must have same number of rows' );
assert( size(b,2) == 1, 'b must be vector' );
alpha = 0.1; % alpha \in (0,1/2)
beta  = 0.8; % beta  \in (0,1)
n = zeros(numel(b),1); % dual variable

x = x0;
[fx,grad,hess] = f(x);
r = [grad(:)+A'*n; A*x-b];

if verbose
    t = 0;
    fprintf('Iter \t f(x) \t\t ||r|| \t\t ||Ax-b|| \t Test\n')
end
as = [];
while norm(A*x-b) > eps || norm(r) > eps
    if length(as) == 3 && min(as) < 1e-3 && as(1) == as(2) && as(1) == as(3)
       break
    end
    if verbose
        t = t+1;
        fprintf('%2.4d \t %2.4d \t %2.4d \t %2.4d \t %2.4d \n',t,fx,norm(r),norm(A*x-b),det(test(x)));
    end
    
    foo = -[hess, A'; A, zeros(numel(b))]\r;
    dx = foo(1:length(x)); % primal Newton step
    dn = foo(length(x)+1:end); % dual Newton step
    r_  = r;
    
    a = 1;
    [fx,grad,hess] = f(x + a*dx);
    if isempty(dn) % If there are no equality constraints
        r = grad(:);
    else
        r = [grad(:)+A'*(n + a*dn); A*(x + a*dx)-b];
    end
    while imag(fx) ~= 0 || test(x + a*dx) < 0 || norm(r) > (1-a*alpha)*norm(r_) % crossed the boundary
        a = a*beta;
        [fx,grad,hess] = f(x + a*dx);
        if isempty(dn)
            r = grad(:);
        else
            r = [grad(:)+A'*(n + a*dn); A*(x + a*dx)-b];
        end
    end
    x = x + a*dx;
    if ~isempty(n)
        n = n + a*dn;
    end
    if length(as) < 3
        as = [as a];
    else
        as = [as(2:3) a];
    end
end
warning('on','MATLAB:nearlySingularMatrix')