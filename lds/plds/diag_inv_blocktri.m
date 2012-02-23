function [diag_iH,iH]=diag_inv_blocktri(H,blksize,n_blks);

%computes diagonal blocks of the inverse of a block-tridiagonal matrix
%---useful for computing variance and sufficient covariance statistics
%uses algorithm from A. Asif and J. M. F. Moura, Block Matrices with
%L-Block Banded Inverse: Inversion Algorithms, IEEE Transactions on Signal Processing, vol. 53, no. 2, pp. 630-42, Feb. 2005.
%http://ieeexplore.ieee.org/iel5/78/30126/01381754.pdf

%H is the input Hessian (inverse covariance) matrix
%blksize is the size of each block
%n_blks is the number of blocks

%diag_iH is a vector of the marginal variances
%iH is the bi-block-diagonal submatrix of inv(H), arranged in two-column form

sh=size(H,1);
if(blksize*n_blks~=sh) error('size(H) must match blksize*n_blks.'); end;

on=1:blksize;
cH=chol(H);
onb=on+(n_blks-1)*blksize;
iH=zeros(blksize*n_blks,2*blksize);
diag_iH=zeros(blksize*n_blks,1);
a=inv(cH(onb,onb)'*cH(onb,onb));
iH(onb,on)=a;
diag_iH(onb)=diag(a);
for(i=n_blks-1:-1:1)
        %if(~rem(i,1000)) disp(i); end;
        onb=on+(i-1)*blksize;
        onc=on+(i)*blksize;
        uo=cH(onb,onb)\cH(onb,onc);
        a=inv(cH(onb,onb)'*cH(onb,onb))+uo*iH(onc,on)*uo';
        iH(onb,on)=a; %this is the diagonal block of the posterior covariance
        diag_iH(onb)=diag(a); %the diagonal of the diagonal block gives the marginal variance
        iH(onb,on+blksize)=-uo*iH(onc,on)'; %this is the off-diagonal block of the posterior cov
end;
%disp('done.')


