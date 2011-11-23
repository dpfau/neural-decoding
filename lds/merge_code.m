% merge my version and Zoubin's version of EM for LDS so I can debug sanely
% Copyright Zoubin Ghahramani 1996, David Pfau 2011
function [A,B,C,Q,R,x0,P0,Mu,LL]=merge_code(X,D,K,cyc,tol,fast)

p=length(X(1,:));
T=length(X(:,1));
tiny=exp(-700);
problem=0;

if nargin<6   fast=0; end;
if nargin<5   tol=0.0001; end;
if nargin<4   cyc=100; end;
if nargin<3   K=2; end;
if nargin<2   D=0; end;

Mu=mean(X);
X=X-ones(T,1)*Mu;

% divide into inputs and outputs

U=X(:,1:D);
Y=X(:,D+1:p);
p=p-D;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize with Linear Regression

fprintf('\nInitializing with Linear Regression...\n');
beta=U\Y; % D x p answer
[t1,t2,t3]=svd(beta'); % p x D
t3=t3';

if K<=D,
    if K<=p,
        C=t1*t2(:,1:K);
    elseif p<=D & K>p,
        C=t1*[t2(:,1:p) 0.1*randn(p,K-p)];
    elseif p>D & K>p,
        C=t1*[t2(1:K,1:K); 0.1*randn(K-p,K)];
    end;
    B=t3(1:K,:);
else
    if p<=D,
        C=t1*[t2(1:p,1:p) 0.1*randn(p,K-p)];
    elseif p>D,
        C=t1*[t2(1:D,1:D) 0.1*randn(D,K-D); 0.1*randn(p-D,K)];
    end;
    B=[t3; 0.1*randn(K-D,D)];
end;

Xhat=U*B';
Yhat=Xhat*C';
Ydiff=Y-Yhat;
R=sum(Ydiff.*Ydiff,1)/T;
A=eye(K);
x0=mean(Xhat)';
t4=cov(Xhat,1);
Q=t4+eye(K)*max(eig(t4))*0.01; % ill conditioned.
P0=Q;

A_ = A; B_ = B; C_ = C; Q_ = Q; R_ = diag(R); x0_ = x0; P0_ = P0; % my copy of all the parameters

clear t1 t2 t3 t4 Xhat Yhat Ydiff beta X;

fprintf('\nInitialized.\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I=eye(K);
Y=Y';
U=U';
lik=0;
LL=[];
const=(2*pi)^(-p/2);

YY=sum(Y.*Y,2)/T;

Xpre=zeros(K,T);   % P(x_t | y_1 ... y_{t-1})
Xcur=zeros(K,T);   % P(x_t | y_1 ... y_t)
Xfin=zeros(K,T);   % P(x_t | y_1 ... y_T)    given all outputs

Ppre=zeros(T*K,K);
Pcur=zeros(T*K,K);
Pfin=zeros(T*K,K); % Var(x_t|{y})
Pt=zeros(T*K,K); % E(x_t x_t | {y})
Pcov=zeros(T*K,K); % Var(x_t,x_t-1|{y}) used to be x_t,x_{t+1}
Pc=zeros(T*K,K); % E(x_t x_t-1 | {y}) used to be x_t,x_{t+1}
Kcur=zeros(T*K,p);
invP=zeros(T*p,p);
J=zeros(T*K,K);

for cycle=1:cyc
    
    % E STEP
    
    %% FORWARD PASS - Zoubin
    
    Xpre(:,1)=x0*ones;
    Ppre(1:K,:)=P0;
    invR=diag(1./(R+(R==0)*tiny));
    
    for t=1:T
        tk=(t-1)*K+1:t*K; tp=(t-1)*p+1:t*p;
        if (K<p)
            T1=invR*C;
            invP(tp,:)=invR-T1*Ppre(tk,:)*inv(I+C'*T1*Ppre(tk,:))*T1';
        else
            invP(tp,:)=inv(diag(R)+C*Ppre(tk,:)*C');
        end;
        Kcur(tk,:)=Ppre(tk,:)*C'*invP(tp,:);
        Xcur(:,t)=Xpre(:,t)+Kcur(tk,:)*(Y(:,t)-C*Xpre(:,t));
        Pcur(tk,:)=Ppre(tk,:)-Kcur(tk,:)*C*Ppre(tk,:);
        if (t<T)
            Xpre(:,t+1)=A*Xcur(:,t)+B*U(:,t);
            Ppre(tk+K,:)=A*Pcur(tk,:)*A'+Q;
        end;
    end;
    
    %% Forward Pass - Me
    
    z_ = zeros(K,T);
    V_ = zeros(K,K,T);
    VV_ = zeros(size(V_));
    P = zeros(size(V_));
    
    ll = zeros(T,1); % log likelihood
    zt = x0_;
    Pt = P0_;
    
    Rinv = R_^-1; % store for fast matrix inversion
    for i = 1:T
        % Precision matrix of y(:,i) given y(:,1:i-1).
        if size(Y,1) > 50
            % Same as (C*Pt*C' + R)^-1 by Woodbury lemma.
            T = Rinv*C_;
            Sinv = Rinv - T*(Pt^-1 + C_'*T)^-1*T';
        else
            Sinv = (C_*Pt*C_' + R_)^-1;
        end
        xt = Y(:,i) - C_*zt; % residual
        ll(i) = - 0.5*( xt'*Sinv*xt + size(Y,1)*log( 2*pi ) - log( det( Sinv ) ) ); % log likelihood of one observation
        
        % update
        Kt = Pt*C_'*Sinv; % Kalman gain
        zt = zt + Kt*xt;
        if i > 1
            VV_(:,:,i) = A_*Vt - Kt*C_*A_*Vt;
        else
            VV_(:,:,i) = A_*P0_ - Kt*C_*A_*P0;
        end
        Vt = Pt - Kt*C_*Pt;
        
        z_(:,i) = zt;
        V_(:,:,i) = Vt;
        
        % predict
        zt = A_*zt;
        zt = zt + B_*U(:,i);
        Pt = A_*Vt*A_' + Q_;
        P(:,:,i) = Pt;
    end
    
    %% BACKWARD PASS - Zoubin
    
    t=T; tk=(t-1)*K+1:t*K;
    
    Xfin(:,t)=Xcur(:,t);
    Pfin(tk,:)=Pcur(tk,:);
    Pt(tk,:)=Pfin(tk,:) + Xfin(:,t)*Xfin(:,t)';
    
    for t=(T-1):-1:1
        tk=(t-1)*K+1:t*K;
        J(tk,:)=Pcur(tk,:)*A'*inv(Ppre(tk+K,:));
        Xfin(:,t)=Xcur(:,t)+J(tk,:)*(Xfin(:,t+1)-Xpre(:,t+1));
        Pfin(tk,:)=Pcur(tk,:)+J(tk,:)*(Pfin(tk+K,:)-Ppre(tk+K,:))*J(tk,:)';
        Pt(tk,:)=Pfin(tk,:) + Xfin(:,t)*Xfin(:,t)'; % E(x_t x_t)
    end;
    
    t=T;
    tk=(t-1)*K+1:t*K;
    Pcov(tk,:)=(I-Kcur(tk,:)*C)*A*Pcur(tk-K,:);
    Pc(tk,:)=Pcov(tk,:)+Xfin(:,t)*Xfin(:,t-1)';
    
    for t=(T-1):-1:2
        tk=(t-1)*K+1:t*K;
        Pcov(tk,:)=Pcur(tk,:)*J(tk-K,:)'+J(tk,:)*(Pcov(tk+K,:)-A*Pcur(tk,:))*J(tk-K,:)';
        Pc(tk,:)=Pcov(tk,:)+Xfin(:,t)*Xfin(:,t-1)';
    end;
    
    %% Backward Pass - me
    
    z = zeros(size(z_));
    V = zeros(size(V_));
    if nargout > 3
        VV = zeros(size(VV_));
    end
    
    zt = z_(:,end);
    Vt = V_(:,:,end);
    
    for i = T-1:-1:1
        z(:,i+1) = zt;
        V(:,:,i+1) = Vt;
        
        Lt = V_(:,:,i)*A_'*P(:,:,i)^-1;
        zpred = A_*z_(:,i);
        zpred = zpred + B_*U(:,i);
        zt = z_(:,i) + Lt*(zt - zpred);
        VV(:,:,i+1) = VV_(:,:,i+1) + (Vt - V_(:,:,i+1))*(V_(:,:,i+1))^-1*VV_(:,:,i+1);
        Vt = V_(:,:,i) + Lt*(Vt - P(:,:,i))*Lt';
    end
    z(:,1) = zt;
    V(:,:,1) = Vt;
    
    %% Calculate Likelihood - Zoubin
    
    oldlik=lik;
    lik=0;
    for t=1:T % Using innovations form
        tp=(t-1)*p+1:t*p;
        MM=invP(tp,:);
        dM=sqrt(det(MM));
        if (isreal(dM) & dM>0)
            Ydiff=Y(:,t)-C*Xpre(:,t);
            lik=lik+log(dM)-0.5*sum(sum(Ydiff.*(MM*Ydiff)));
        else
            problem=1;
        end;
    end;
    if problem
        fprintf(' problem '); problem=0;
    end;
    
    lik=lik+T*log(const);
    LL=[LL lik];
    
    fprintf('cycle %g lik %d',cycle,lik);
    
    if (cycle<=2)
        likbase=lik;
    elseif (lik<oldlik)
        fprintf(' violation');
    elseif ((lik-likbase)<(1 + tol)*(oldlik-likbase)|isinf(lik))
        fprintf('\n');
        break;
    end;
    fprintf('\n');
    
    %% M STEP - Zoubin
    
    % Re-estimate A,B,C,Q,R,x0,P0;
    
    x0old=x0;
    x0=Xfin(:,1);
    T1=Xfin(:,1)'-mean(Xfin(:,1)');
    P0=Pfin(1:K,:)+T1'*T1;
    
    Ptsum=zeros(K);
    for i=1:T
        tk=(i-1)*K+1:i*K;
        Ptsum=Ptsum+Pt(tk,:);
    end;
    
    YX=Y*Xfin';
    C=YX*pinv(Ptsum);
    
    R=YY-diag(C*YX')/T;
    
    PC1=zeros(K);
    PT1=zeros(K);
    PT2=zeros(K);
    UU=zeros(D);
    UX1=zeros(D,K);
    UX2=zeros(D,K);
    for t=2:T
        tk=(t-1)*K+1:t*K;
        UX1=UX1+U(:,t-1)*Xfin(:,t-1)';
        UX2=UX2+U(:,t-1)*Xfin(:,t)';
        PC1=PC1+Pc(tk,:); %P_{t,t-1}
        PT1=PT1+Pt(tk,:);
        PT2=PT2+Pt(tk-K,:);
        UU=UU+U(:,t-1)*U(:,t-1)';
    end;
    
    T1=[PT2 UX1'; UX1 UU];
    T2=[PC1 UX2'];
    T3=T2*inv(T1);
    A=T3(:,1:K);
    B=T3(:,K+1:K+D);
    
    Q1=PT1-A*PC1'-B*UX2;
    Q=diag(diag(Q1))/(T-1);
    
    if (det(Q)<0 | sum(sum(abs(Q-Q')))>0.001)
        fprintf('Q problem\n');
    end;
    Q=(Q+Q')/2;
    
    %% M step - me
    
    x0_ = z(:,1);
    P0_ = V(:,:,1) + (z(:,1)-mean(z(:,1)))*(z(:,1)-mean(z(:,1)))';
    
    Ptt1  = sum(VV(:,:,2:end),3) + z(:,2:end)*z(:,1:end-1)';
    Pt1t1 = sum(V(:,:,1:end-1),3) + z(:,1:end-1)*z(:,1:end-1)';
    Ptt   = sum(V(:,:,2:end),3) + z(:,2:end)*z(:,2:end)';
    
    u2 = U(:,1:end-1); z1 = z(:,1:end-1); z2 = z(:,2:end);
    AB = [Ptt1, z2*u2']*[Pt1t1, z1*u2'; u2*z1', u2*u2']^-1;
    A_ = AB(:,1:K);
    B_ = AB(:,K+1:end);
    Q_ = 1/(T-1)*(Ptt - A*Ptt1' - B*u2*z2');
    Q_ = diag(diag(Q_)); % as the model is nonidentifiable, might as well force Q to be diagonal (following Zoubin)
    
    C_ = (Y*z')*(sum(V,3) + z*z')^-1;
    R_ = 1/T*(Y*Y' - C*z*Y');
    R_ = diag(diag(R_));
    
    fprintf('%d\n',max(max(abs(z_-Xcur))))
    fprintf('%d\n',max(max(abs(z-Xfin))))
    fprintf('%d\n',max(max(abs(x0_-x0))))
    fprintf('%d\n',max(max(abs(P0_-P0))))
    fprintf('%d\n',max(max(abs(A_-A))))
    fprintf('%d\n',max(max(abs(B_-B))))
    fprintf('%d\n',max(max(abs(C_-C))))
    fprintf('%d\n',max(max(abs(Q_-Q))))
    fprintf('%d\n',max(max(abs(R_-diag(R)))))
end;


