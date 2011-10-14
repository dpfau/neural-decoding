function test(n,m,l)

A = randn(n,n)/10;
B = randn(n,m);
C = randn(l,n);
D = randn(l,m);
X0 = randn(n,1);
u = randn(m,1024);

foo = rand(n+l);
E = 0.0001*foo*foo';
Q = E(1:n,1:n);
R = E(n+1:end,n+1:end);
S = E(1:n,n+1:end);

[~,y] = gen(A,B,C,D,X0,u,Q,R,S);

[a,b,c,d,x0] = moesp(y,u,16,512,.7);
[aa,bb,cc,dd,x,q,r,s] = n4sid(y,u,16,.7);

[~,y_] = gen(a,b,c,d,x0,u,[],[],[]);
[~,y__] = gen(aa,bb,cc,dd,x(:,1),u(:,9:end),[],[],[]);

for i = 1:size(y,1)
    subplot(size(y,1),1,i);
    plot(1:size(y,2),y(i,:),1:size(y_,2),y_(i,:),8+(1:size(y__,2)),y__(i,:))
end
legend('Orig','moesp','n4sid')