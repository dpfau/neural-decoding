function [y fy] = lbfgs(obj,m,tol,iter,display)
% Limited memory BFGS method for nonlinear function optimization
% Usage:
%   [y fy] = lbfgs(obj,m,tol,iter,display)
%   obj - An object containing all parameters of the function to be optimized
%         obj.f - the objective function for which we find the minimum 
%                 value y. Returns value and gradient
%         obj.A - the matrix of linear inequality constraints for the
%                 parameters, obj.A*params > obj.p
%         obj.p - the vector part of the linear inequality constraints

if nargin < 5
    display = 0;
end
if nargin < 4
    iter = Inf;
end
c1 = 1e-4;
c2 = 0.9; % Following the totally arbitrary numbers in Nocedal and Wright

k = 0;
s = [];
z = [];
grad = Inf;
if display, fprintf('Iter \t f(y) \t\t ||grad|| \n'), end
while norm(grad) > tol && k < iter % L-BFGS, algorithm 9.2 in Nocedal and Wright
    y = obj.get_params();
    [fy grad] = obj.f();
    if display, fprintf('%i \t %2.4d \t %2.4d \n',k,fy,norm(grad)), end
    
    % Two loop recursion, algorithm 9.1 in Nocedal and Wright
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
    
    % Backtracking line search, algorithm 3.2 in Nocedal and Wright
    p = -r;
    a0 = 0;
    if any( obj.A*p < 0 ) % If following the line search far enough out takes us outside the inequality constraints
        bound = -(obj.A*y - obj.p)./(obj.A*p);
        a_ = min(bound(bound>0));
        a1 = min(a_/2,1);
    else
        a_ = 100;
        a1 = 1;
    end
    dfy = grad'*p;
    fy0 = fy;
    i = 1;
    while 1
        [fy1 grad1] = obj.f(y+a1*p);
        if fy1 > fy + c1*a1*dfy || ( fy1 >= fy0 && i > 1 ) 
            while 1 % "zoom" f'n, algorithm 3.3 in Nocedal and Wright
                a = -dfy*a1^2/(2*(fy1 - fy - dfy*a1)); % Quadratic interpolation, eq'n 3.42 in Nocedal and Wright
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
            fy_   = fy1;
            grad_ = grad1;
            break
        end
        if grad1'*p >= 0
            while 1 % "zoom" f'n, algorithm 3.3 in Nocedal and Wright
                a = dfy*a1^2/(2*(fy1 - fy - dfy*a1)); % Quadratic interpolation, eq'n 3.42 in Nocedal and Wright
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
        a0 = a1;
        a1 = (a0+a_)/2;
        i = i+1;
        fy0 = fy1;
    end
    
    % Update parameters
    s = [s, a*p];
    z = [z, grad_-grad];
    obj = obj.set_params(y+a*p);
    k = k+1;
    if k > m
        s = s(:,2:end);
        z = z(:,2:end);
    end
    fy = fy_;
    grad = grad_;
end