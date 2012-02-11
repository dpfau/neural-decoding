function y = sinct(x,k,T)
% Interpolant function sin(pi*x).*cot(pi*x/T)/T, which near 0 is almost
% sinc(x) = sin(pi*x)./(pi*x), and its first and second derivative

if nargin == 2
    T = size(x,1);
end
switch k
    case 0
        y = sin(pi*x).*cot(pi*x/T)/T;
        y(mod(x,T)==0) = 1;
    case 1
        y = pi/T  *cos(pi*x).*cot(pi*x/T) - ...
            pi/T^2*sin(pi*x).*csc(pi*x/T).^2;
        y(mod(x,T)<1e-7 | mod(x,T)-T>-1e-7) = 0;
    case 2
        y = -2*pi^2/T^2*cos(pi*x).*csc(pi*x/T).^2 - ...
            pi^2/T     *sin(pi*x).*cot(pi*x/T)    + ...
            2*pi^2/T^3 *sin(pi*x).*cot(pi*x/T).*csc(pi*x/T).^2;
        y(mod(x,T)<1e-7 | mod(x,T)-T>-1e-7) = -pi;
    otherwise
        error('Only zeroth, first and second derivative allowed.')
end