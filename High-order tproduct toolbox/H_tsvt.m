function [X,htnn,tsvd_rank] = H_tsvt(Y,rho,transform)

%
% Order-d (d>=3) Tensor Singular Value Thresholding (T-SVT) operator, 
% which solves the order-d (d>=3) TNN minimization problem (the proximal operator of the TNN under linear transform)
%
% Input:
%       rho     -   a constant: rho > 0
%       Y       -   n1*n2*n3*бнбн*nd tensor
%   transform   -   a structure which defines the linear transform
%       transform.L: the linear transform of two types:
%                  - type I: function handle, i.e., @fft, @dct
%                  - type II: invertible  transform  matrix of size ni*ni, for i=3,бнбн,d
%
%       transform.inverseL: the inverse linear transform of transform.L
%                         - type I: function handle, i.e., @ifft, @idct
%                         - type II: inverse  transform  matrix of transform.L
%
%       transform.rho: a constant which indicates whether the following property holds for the linear transform or not:
%                    U_3'*U_3=U_3*U_3'=l3*I,бн, U_d'*U_d=U_d*U_d'=ld*I, for some l3>0,бн,ld>0
%                    let rho=l3*бн*ld
%                  - transform.rho > 0: indicates that the above property holds. Then we set transform.rho = rho.
%                  - transform.rho < 0: indicates that the above property does not hold. Then we can set transform.rho = c, for any constant c < 0.
%
%       If not specified, fft is the default transform, i.e.,
%       transform.L = @fft, transform.rho = n3*бнбн*nd, transform.inverseL = @ifft. 
%
% Output:
%        X       -   the solution of tensor singular value thresholding
%        htnn    -   tensor nuclear norm of order-d tensor X
%   tsvd_rank    -   t-SVD rank  of  order-d tensor X
%
% References:
% Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University. 
% June, 2018. https://github.com/canyilu/tproduct.
%
% Canyi Lu, Tensor-Tensor Product Toolbox 2.0. Carnegie Mellon University. 
% April, 2021. https://github.com/canyilu/Tensor-tensor-product-toolbox.
%
% Wenjin Qin, Hailin Wang, Feng Zhang, Jianjun Wang, Xin Luo, Tingwen Huang. 
% Low-Rank High-Order Tensor Completion with Applications in Visual Data [J]. 
% IEEE Transactions on Image Processing, 2022, 31: 2433-2448.
% ResearchGate: https://www.researchgate.net/publication/359116039_Low-Rank_High-Order_Tensor_Completion_With_Applications_in_Visual_Data
%
% Wenjin Qin, Hailin Wang, Weijun Ma, Jianjun Wang. Robust high-order tensor recovery via nonconvex low-rank approximation[C]. 
% In: Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
% 2022: 3633-3637.
%  ResearchGate: https://www.researchgate.net/publication/360423337_Robust_High-Order_Tensor_Recovery_Via_Nonconvex_Low-Rank_Approximation
%
% Written by  Wenjin Qin  (qinwenjin2021@163.com)
%

Ndim = length(size(Y));
Nway=size(Y);
if nargin == 3
     if transform.rho < 0
         error("property U_3'*U_3=U_3*U_3'=l3*I,бн, U_d'*U_d=U_d*U_d'=ld*I does not holds for some l3>0,бн,ld>0.");
     end
else
    % fft is the default transform
    transform.L = 'fft'; transform.rho= prod(Nway(3:end)); transform.inverseL = 'ifft';
end

htnn = 0;
tsvd_rank = 0;
X = zeros(Nway);
if strcmp(transform.L,'fft')
    %[X,htnn,tsvd_rank]=prox_htnn_fft(Y,rho);
    [X,htnn,tsvd_rank]=H_tsvt_fft(Y,rho);
    %
else
    for x = 3:Ndim
        Y = tmprod(Y,transform.L{x-2},x);
    end
    
    for i=1:prod(Nway(3:end))
        [U,S,V] = svd(Y(:,:,i),'econ');
        S = diag(S);
        r = length(find(S>rho));
        if r>=1
            S =max( S(1:r)-rho,0);
            X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
            htnn = htnn+sum(S);
            tsvd_rank = max(tsvd_rank,r);
        end
    end
    htnn = htnn/transform.rho;
    
    for y = Ndim:-1:3
        X = tmprod(X,inv(transform.L{y-2}),y);
    end  
end


function [X,htnn,tsvd_rank] = H_tsvt_fft(Y,rho)
%
% The proximal operator for the order-D tensor nuclear norm under FFT
%
% Written by  Wenjin Qin  (qinwenjin2021@163.com)
%

Ndim = length(size(Y));
Nway=size(Y);
htnn = 0;
tsvd_rank = 0;
X = zeros(Nway);
min12=min(Nway(1),Nway(2));
s=zeros([min12, min12, Nway(3:end)]);

L = ones(1,Ndim);
for x = 3:Ndim
    Y = fft(Y,[],x);
    L(x) = L(x-1) * Nway(x);
end

[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S);
r = length(find(S>rho));
%if r>=1
if r>=1
    S = max(S(1:r)-rho,0);
    X(:,:,1) = U(:,1:r)*diag(S)*V(:,1:r)';
    s(:,:,1)=diag([S' zeros(1,min12-r)]);
    htnn = htnn+sum(diag(s(:,:,1)));
    tsvd_rank = max(tsvd_rank,r);
end

for j = 3 : Ndim
    for i = L(j-1)+1 : L(j)
        I = unfoldi(i,j,L);
        halfnj = floor(Nway(j)/2)+1;
        if I(j) <= halfnj && I(j) >= 2
            [U,S,V] = svd(Y(:,:,i),'econ');
            S = diag(S);
            r = length(find(S>rho));
            if r>=1
                S = max(S(1:r)-rho,0);
                X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
                s(:,:,i)=diag([S' zeros(1,min12-r)]);
                htnn = htnn+sum(diag(s(:,:,i)));
                tsvd_rank = max(tsvd_rank,r);
            end       
        elseif I(j) > halfnj
            n_ = nc(I,j,Nway);
            i_ = foldi(n_,j,L);
            X(:,:,i) = conj( X(:,:,i_));
            s(:,:,i)=conj( s(:,:,i_));
            htnn = htnn+sum(diag(s(:,:,i)));
        end
    end
end
htnn = htnn/prod(Nway(3:end));

for z = Ndim:-1:3
    X = (ifft(X,[],z));
end
X = real(X);







