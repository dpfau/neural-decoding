function [y fy] = lbfgs(obj,m,tol,iter,display)

if nargin < 5
    display = 0;
end
if nargin < 4
    iter = Inf;
end
c1 = 1e-4;
c2 = 0.9; % Following the totally arbitrary numbers in Nocedal and Wright

y = obj.get_params();
[fy grad] = obj.f();
k = 0;
s = [];
z = [];
if display, fprintf('Iter \t f(y) \t ||grad||'), end
while norm(grad) > tol && k < iter
    if display, fprintf('%i \t %2.4d \t %2.4d \n',k,fy,norm(grad)), end
    % Two loop recursion
    a = zeros(1,min(m,k));
    q = grad;
    for i = min(m,k):-1:1
        a(i) = s(:,i)'*q/(z(:,i)'*s(:,i));
        q = q - a(i)*z(:,i);
    end
    if k == 0
        r = q;
    else
        r = (s(:,end)'*z(:,end))/(z(:,end)'*z(:,end))*q;
    end
    for i = 1:min(m,k)
        b = z(:,i)'*r/(z(:,i)'*s(:,i));
        r = r + s(:,i)*(a(i)-b);
    end
    
    % Backtracking line search
    a0 = 0;
    a1 = 1;
    a_ = 100;
    p = -r;
    dfy = grad'*p;
    fy0 = fy;
    i = 1;
    while 1
        [fy1 grad1] = obj.f(y+a1*p);
        if fy1 > fy + c1*a1*dfy || ( fy1 >= fy0 && i > 1 )
            while 1
                a = interpolate; % Need to define this!
                [fy_ grad_] = obj.f(y+a*p);
                dfy_ = grad_'*p;
                if fy_ > fy + c1*a*dfy || fy_ >= fy0
                    a1 = a;
                else
                    if abs(dfy_) <= -c2*dfy
                        break
                    elseif dfy_*(a1-a0) >= 0
                        a1 = a0;
                    end
                    a0 = a;    
                end
            end
            break
        end
        if abs(grad1'*p) <= -c2*grad'*p
            a = a1;
            break
        end
        if grad1'*p >= 0
            a = zoom(a0,a1,obj,y,p,c1,c2);
            break
        end
        a0 = a1;
        a1 = mean(a0,a_);
        i = i+1;
        fy0 = fy1;
    end
    
    % Update parameters
    s = [s, a*p];
    z = [z, grad_-grad];
    obj.set_params(y+a*p);
    k = k+1;
    if k > m
        s = s(:,2:end);
        z = z(:,2:end);
    end
    fy = fy_;
    grad = grad_;
end