% Example from Verhaegen and Verdault of biased estimation in the presence
% of colored output noise

A = [1.5, 1.0; -0.7, 0];
B = [1.0; 1.5];
C = [1.0, 0];
D = 0;
X0 = [0; 0];

n = 50;

eigs = zeros(2,n);

for i = 1:n
    u = randn(1,1000);
    e = sqrt(0.2)*randn(1,1001);
    [x,y] = gen(A,B,C,D,X0,u,[],[],[]);
    y = y + (e(2:end) + 0.5*e(1:end-1))./(1-1.69*e(2:end)+0.96*e(1:end-1));
    a = moesp(y,u,10,1000,'orth_svd',-2);
    aa = moesp(y,u,10,1000,'oblique',-2);
    foo = eig(a);
    bar = eig(aa);
    eigs(:,i) = [foo(1); bar(1)];
    disp(int2str(i))
    clf
    scatter(real(eigs(1,:)),imag(eigs(1,:)))
    hold on
    scatter(real(eigs(2,:)),imag(eigs(2,:)),'r')
end