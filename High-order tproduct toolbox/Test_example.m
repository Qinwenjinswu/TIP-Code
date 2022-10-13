%% H_bdiag, H_bunfold, H_bcirc, H_b_toeplitz,  H_b_hankel
clear all
a=randn(5,5,10,10);
a1=H_bdiag(a);
a2=H_bunfold(a);
a3=H_bcirc(a);
a4=H_b_toeplitz(a);
a5= H_b_hankel(a);

%%  H_bcirc(A) can be block diagonalized via the Discrete Fourier Transformation (DFT) 
clear all;
addpath(genpath(cd));
a=randn(5,5,8,8,8);
F_3=dftmtx(8);
F_4=dftmtx(8);
F_5=dftmtx(8);
F_i3=inv(F_3);
F_i4=inv(F_4);
F_i5=inv(F_5);

F= kron(F_5,kron(F_4,F_3));
F_hat= kron(F_i5,kron(F_i4,F_i3));
e1=eye(5,5);
e2=eye(5,5);

bcirc_result=kron(F,e1)*H_bcirc(a)*kron(F_hat,e2);
for i=3:5
    a=fft(a,[],i);
end

bdig_result=H_bdiag(a);
bdig_result(11,11)
bcirc_result(11,11)

%% lineartransform(A,transform),  inverselineartransform(A,transform)
clear all;
a=randn(5,5,15,15);
b=a;
transform.L='dct';
x1=inverselineartransform(a,transform);
for y=4:-1:3
    a=idct(a,[],y);
end
a==x1;

clear all;
a=randn(5,5,15,15);
b=a;
transform.L='dct';
x1=lineartransform(b,transform);
for y=3:4
    b=dct(b,[],y);
end
x1==b;

%% H_tprod(A,B,transform)
clear all;
addpath(genpath(cd));
a=randn(4,4,14,14);
b=randn(4,4,14,14);
transform.L='fft';
U{1}=dctmtx(14);U{2}=dctmtx(14);
transform.L=U;
c=H_tprod(a,b,transform);

clear all;
a=randn(15,15,81,80);
a1=randn(15,15,81,80);
transform.L='fft';
b=H_tprod(a,a1,transform);

c1=lineartransform(a,transform);
c2=lineartransform(a1,transform);
c3=zeros(15,15,81,80);
for i=1:81*80
   c3(:,:,i)=c1(:,:,i)*c2(:,:,i);
end
c4=inverselineartransform(c3,transform);
c4=real(c4);
k1=b(:,:,50);
k2=c4(:,:,50);


%% H_teye(Nway,transform)
clear all
addpath(genpath(cd));
transform.L='dct';
%transform.L='fft';
a=lineartransform(H_teye([5 5 15 15 ],transform),transform);

clear all
a=[5,5,6,6,6];
transform.L='fft';
a1=H_teye(a,transform);
a2 = lineartransform(a1,transform);

%% H_tinv(A,transform)
clear all;
addpath(genpath(cd))
a=randn(4,4,4,4);
%transform.L='fft';
  U{1}=dctmtx(4);U{2}=dctmtx(4);
  transform.L=U;
b=H_tinv(a,transform);
c=H_tprod(a,b,transform);
x1=lineartransform(c,transform);



%% H_grt(NwayA,transform)
clear all
Nway=[5 5 4 6];
transform.L='fft';
a=H_grt(Nway,transform);
a1 = lineartransform(a,transform);

clear all
a=zeros(5,5,6,6,6);
a(:,:,1)=randn(5,5);
transform.L='fft';
a1 = lineartransform(a,transform);

clear all
a=[5,5,6,6,6];
transform.L='fft';
a1=H_grt(a,transform);
a2 = lineartransform(a1,transform);

%% H_trank(X,transform,tol)
clear all;
r=15;
a=randn(20,r,40,40);
b=randn(r,20,40,40);
transform.L='fft';
c=H_tprod(a,b);
H_trank(c)
%
U{1}=dctmtx(40);U{2}=dctmtx(40);
transform.L=U;
transform.rho=1;
c1=H_tprod(a,b,transform);
H_trank(c1,transform)


%% H_tsvd(X,transform,opt)
%% case 1: skinny
clear all
transform.L='fft';
opt='skinny';

a=randn(60,5,40,40);
b=randn(5,20,40,40);
c=H_tprod(a,b,transform);
H_trank(c)
[U1 ,S1 ,V1]=H_tsvd(c,transform,opt);

U{1}=dctmtx(40);U{2}=dctmtx(40);
transform.L=U;
transform.rho=1;
c=H_tprod(a,b,transform);
H_trank(c,transform);
[U1 ,S1 ,V1]=H_tsvd(c,transform,opt);

%% case 2: econ
clear
addpath(genpath(cd))
a=randn(18,15,41,41);
opt='econ';
transform.L='fft';
[u1 ,s1 ,v1]=H_tsvd(a,transform,opt);


u2=zeros(18,15,41,41);
s2=zeros(15,15,41,41);
v2=zeros(15,15,41,41);
%al=lineartransform(a,transform);
for x = 3:4
        a = fft(a,[],x);
end
 
for i=1:41*41
   [u2(:,:,i) ,s2(:,:,i),v2(:,:,i)]=svd(a(:,:,i),'econ');
end
% u2=inverselineartransform(u2,transform);
% s2=inverselineartransform(s2,transform);
% v2=inverselineartransform(v2,transform);
 for jj =4:-1:3
        u2 = ifft(u2,[],jj);
        s2 = ifft(s2,[],jj);
        v2 = ifft(v2,[],jj);
 end
u2=real(u2); s2=real(s2); v2=real(v2);

k1=u1(:,:,1,3);
k2=u2(:,:,1,3);

%% case 3: full
%%
clear
addpath(genpath(cd))
a=randn(18,15,4,4);
opt='full';
 U{1}=dctmtx(4);U{2}=dctmtx(4);
 transform.L=U;
 transform.L='fft';
[u1 ,s1 ,v1]=H_tsvd(a,transform,opt);

 u2=zeros(18,18,4,4);
 s2=zeros(18,15,4,4);
 v2=zeros(15,15,4,4);
% u2=zeros(18,15,4,4);
% s2=zeros(15,15,4,4);
% v2=zeros(15,15,4,4);
%al=lineartransform(a,transform);
for x = 3:4
        a = fft(a,[],x);
end
 
for i=1:16
   [u2(:,:,i) ,s2(:,:,i),v2(:,:,i)]=svd(a(:,:,i));
end
% u2=inverselineartransform(u2,transform);
% s2=inverselineartransform(s2,transform);
% v2=inverselineartransform(v2,transform);
 for jj =4:-1:3
        u2 = ifft(u2,[],jj);
        s2 = ifft(s2,[],jj);
        v2 = ifft(v2,[],jj);
 end
 u2=real(u2); s2=real(s2); v2=real(v2);
k1=u1(:,:,1,3);
k2=u2(:,:,1,3);



%% H_tqr(X,transform,opt)
%% case 1: econ
clear
a=randn(15,5,10,10);
b=randn(5,15,10,10);
transform.L='fft';
opt='econ';
c=H_tprod(a,b,transform);
[q ,r]=H_tqr(c,transform,opt);

U{1}=dctmtx(10);U{2}=dctmtx(10);
transform.L=U;
opt='econ';
[q ,r]=H_tqr(c,transform,opt);

c=H_tprod(q,H_tran(q,transform),transform);
c1 = lineartransform(c,transform);
%%
clear
addpath(genpath(cd))
a=randn(40,15,80,100);
opt='econ';
transform.L='fft';
tic
[q1 ,r1]=H_tqr(a,transform,opt);
toc

q2=zeros(40,15,80,100);
r2=zeros(15,15,80,100);

tic
al=lineartransform(a,transform);
for i=1:100*80
   [q2(:,:,i) ,r2(:,:,i)]=qr(al(:,:,i),0);
end
q2=inverselineartransform(q2,transform);
r2=inverselineartransform(r2,transform);
toc

q2=real(q2); r2=real(r2);

k1=q1(:,:,210);
k2=q2(:,:,210);
%% case 2: full
clear all;
a=randn(6,6,4,4,4);
opt='full';
transform.L='fft';
[q1 ,r1]=H_tqr(a,transform,opt);
c1=H_tprod(q1,H_tran(q1));
c2 = lineartransform(c1);
c2(:,:,18);

clear all;
a=randn(10,8,4,4);
opt='full';
U{1}=dctmtx(4);U{2}=dctmtx(4);
transform.L=U;
[q1 ,r1]=H_tqr(a,transform,opt);
c1=H_tprod(q1,H_tran(q1,transform),transform);
c2 = lineartransform(c1,transform);

%%
clear
a=randn(8,5,10,10);
U{1}=dctmtx(10);U{2}=dctmtx(10);
transform.L=U;
opt='full';
[q ,r]=H_tqr(a,transform,opt);
c=H_tprod(q,H_tran(q,transform),transform);
c1 = lineartransform(c,transform);

%% H_rtsvd(X,transform,k,p,power_iter)
clear all;
a=randn(50,10,50,50);
b=randn(10,50,50,50);
opt='full';
%transform.L='fft';
U{1}=dctmtx(50);U{2}=dctmtx(50);
transform.L=U;
transform.rho=1;
c=H_tprod(a,b,transform);

tic
[U1 ,S1 ,V1]=H_tsvd(c,transform,opt);
toc

tic
[U2 ,S2 ,V2]=H_rtsvd(c,transform,10,2,2);
toc

a1=H_tprod(U1,S1,transform);
a2=H_tprod(a1,H_tran(V1,transform),transform);

b1=H_tprod(U2,S2,transform);
b2=H_tprod(b1,H_tran(V2,transform),transform);

norm(a2(:)-c(:),2)/norm(c(:),2)
norm(b2(:)-c(:),2)/norm(c(:),2)

H_trank(c,transform)

%% H_eigen(X,transform)
clear all;
addpath(genpath(cd))
a=randn(4,4,10,10);
U{1}=dctmtx(10);
U{2}=dctmtx(10);
U{3}=dctmtx(10);
% U{1}=RandOrthMat(10);
% U{2}=RandOrthMat(10);
% U{3}=RandOrthMat(10);
%  U{1}=dftmtx(10);
%  U{2}=dftmtx(10);
%  U{3}=dftmtx(10);
transform.L='fft';
transform.L=U;
[U, S]=H_eigen(a,transform);
c= H_tprod(U,S,transform);
d=H_tprod(c, H_tinv(U,transform), transform);


clear all;
addpath(genpath(cd))
a=randn(4,4,10,10);
transform.L='fft';
a1=lineartransform(a,transform);
U=zeros(4,4,10,10);
S=zeros(4,4,10,10);
for i=1:100
     [U(:,:,i),S(:,:,i)] = eig(a1(:,:,i));
end
U=inverselineartransform(U,transform);
S=inverselineartransform(S,transform);
 

transform.L='fft';
[U1, S1]=H_eigen(a,transform);

%%  H_tsvt(Y,rho,transform)
clear all
addpath(genpath(cd))
Y=randn(10,10,21,11);
rho=0.5;
U{1}=dctmtx(21);U{2}=dctmtx(11);
transform.L=U;
transform.rho=1;
[X,htnn,tsvd_rank] = H_tsvt(Y,rho,transform);
 H_tnn(X,transform)



%% H_tnn(A,transform)
clear all;
a=randn(5,5,150,40);
transform.L='fft';
Nway=size(a);
transform.rho=prod(Nway(3:end));
a1=H_tnn(a,transform)

htnn=0;
A = lineartransform(a,transform);
for i = 1 : prod(Nway(3:end))
    s = svd(A(:,:,i),'econ');
    htnn = htnn+sum(s);
end
htnn = htnn/transform.rho




%% H_appro(A,k,transform,opt)
clear all
addpath(genpath(cd))
 U{1}=dctmtx(40);U{2}=dctmtx(40);
 transform.L=U;
 transform.L='fft';
a=randn(60,15,40,40);
b=randn(15,20,40,40);
c=H_tprod(a,b,transform);
c1=H_appro(c,15,transform);
norm(c1(:)-c(:))/norm(c(:));

%% 
clear all;
addpath(genpath(cd))
a=randn(5,5,10,10);
b=randn(5,5,10,10);
U{1}=dctmtx(10);
U{2}=dctmtx(10);
transform.L=U;
%transform.L='fft';
a1=lineartransform(a,transform);
b1=lineartransform(b,transform);

c1=norm(a(:))
c=H_bdiag(a1);
c2=norm(c(:))/1

 tr1=0;
 g1=H_bdiag(a1);
 g2=H_bdiag(b1);
 
 tr2=trace(g1*g2');
 
for x=1:100
    tr1=tr1+trace(a(:,:,x)*b(:,:,x)');
end


d1=dot(a(:),b(:));
d2=dot(a1(:),b1(:));

%%
clear all;
addpath(genpath(cd))
 [h, g, a, info] = wfilt_db(30);


