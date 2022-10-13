function [U ,S ,V] = H_tsvd(X,transform,opt)

%
%
% Order-d (d>=3) tensors Singular Value Decomposition (t-SVD)  under any invertible linear transform
%
%
% [U,S,V] = H_tsvd(X,transform,opt) computes the t-SVD under linear transform, i.e., X=U * S *  H_tran(V), 
%where S is a f-diagonal tensor, U and V are orthogonal under linear transform.
%
% Input:
%       X       -   n1*n2*n3*бнбн*nd tensor
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
%
%       opt     -   options for different outputs of U, S and V:
%                   'full': (default) produces full tensor SVD, i.e., X = U * S * H_tran(V), where
%                       U - n1*n1*n3*бнбн*nd
%                       S - n1*n2*n3*бнбн*nd
%                       V - n2*n2*n3*бнбн*nd
%                   'econ': produces the "economy size" decomposition. 
%                       Let m = min(n1,n2). Then, X = U * S * H_tran(V), where
%                       U - n1*m*n3*бнбн*nd
%                       S - m*m*n3*бнбн*nd
%                       V - n2*m*n3*бнбн*nd
%                   'skinny': produces the skinny tensor SVD.
%                       Let r be the t-SVD rank of X. Then, X = U * S * H_tran(V), where
%                       U - n1*r*n3*бнбн*nd
%                       S - r*r*n3*бнбн*nd
%                       V - n2*r*n3*бнбн*nd
%
%
% Output: U, S, V
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

Ndim = length(size(X));
Nway= size(X);
if nargin < 3
    opt = 'econ';
end
if nargin < 2
    % fft is the default transform
    transform.L = 'fft'; transform.rho = prod(Nway(3:end)); transform.inverseL = 'ifft';
end

if strcmp(transform.L,'fft')
    [U ,S, V]=H_tsvd_fft(X,opt);
else
    
    for x = 3:Ndim
        %
        X = tmprod(X,transform.L{x-2},x);
    end
    
    n1=size(X,1);
    n2=size(X,2);
    if strcmp(opt,'skinny') == 1 || strcmp(opt,'econ') == 1
        min12 = min(n1,n2);
        s = zeros(min(Nway(1),Nway(2)),1);
        U= zeros([n1,min12,Nway(3:end)]);
        S = zeros([min12,min12,Nway(3:end)]);
        V = zeros([n2,min12,Nway(3:end)]);
        for i = 1 :prod(Nway(3:end))
            [U(:,:,i),S(:,:,i),V(:,:,i)] = svd(X(:,:,i),'econ');
            s=s+diag(S(:,:,i));
        end
        
        if strcmp(opt,'skinny') == 1
            if transform.rho > 0
                s1 = s/prod(Nway(3:end))/transform.rho;
            else
                % the above property does not hold for transform L
                transformrho=0;
                for x=3:Ndim
                    transformrho=transformrho+norm(transform.L{x-2})^2;
                end
                s1 = s/prod(Nway(3:end))/transformrho;
            end
            tol = 1e-10;
            trank = sum(s1 > tol); % t-SVD rank
            
            U = U(:,1:trank,:);
            U=reshape(U,[n1,trank,Nway(3:end)]);
            V = V(:,1:trank,:);
            V=reshape(V,[n2,trank,Nway(3:end)]);
            S = S(1:trank,1:trank,:);
            S=reshape(S,[trank,trank,Nway(3:end)]);
        end
        
    elseif strcmp(opt,'full') == 1
        U = zeros([n1,n1,Nway(3:end)]);
        S = zeros([n1,n2,Nway(3:end)]);
        V = zeros([n2,n2,Nway(3:end)]);
        for i = 1 :prod(Nway(3:end))
            [U(:,:,i),S(:,:,i),V(:,:,i)] = svd(X(:,:,i));
        end
    end
    
    for y = Ndim:-1:3
        U = tmprod(U,inv(transform.L{y-2}),y);
        S = tmprod(S,inv(transform.L{y-2}),y);
        V = tmprod(V,inv(transform.L{y-2}),y);
    end
    
end

end


function [U,S,V] = H_tsvd_fft(X,opt)
%
% Order-d  tensors Singular Value Decomposition  under FFT
% Written by  Wenjin Qin  (qinwenjin2021@163.com)
%
Ndim = length(size(X));
Nway= size(X);
% if nargin < 2
%      opt = 'econ';
% end

L = ones(1,Ndim);
for x = 3:Ndim
    X = fft(X,[],x);
    L(x) = L(x-1) * Nway(x);
end

n1=size(X,1);
n2=size(X,2);
if strcmp(opt,'skinny') == 1 || strcmp(opt,'econ') == 1
    min12 = min(n1,n2);
    s = zeros(min(Nway(1),Nway(2)),1);
    U= zeros([n1,min12,Nway(3:end)]);
    S = zeros([min12,min12,Nway(3:end)]);
    V = zeros([n2,min12,Nway(3:end)]);

    [U(:,:,1),S(:,:,1),V(:,:,1)] = svd(X(:,:,1),'econ');
    s=s+diag(S(:,:,1));
    
    for j = 3 : Ndim
        for i = L(j-1)+1 : L(j)

            I = unfoldi(i,j,L);
            halfnj = floor(Nway(j)/2)+1;
            if I(j) <= halfnj && I(j) >= 2
                [U(:,:,i),S(:,:,i),V(:,:,i)] = svd(X(:,:,i),'econ');
                %s = s + diag(S(:,:,i))*2;
                s = s + diag(S(:,:,i));
            elseif I(j) > halfnj
                n_ = nc(I,j,Nway);
                i_ = foldi(n_,j,L);
                U(:,:,i) = conj(U(:,:,i_));
                V(:,:,i) = conj(V(:,:,i_));
                S(:,:,i) = conj(S(:,:,i_));
                s = s + diag(S(:,:,i));
            end
            
        end
    end

    if strcmp(opt,'skinny') == 1
        %
        s1 = s/prod(Nway(3:end))^2;
        tol = max(n1,n2)*eps(max(s1));
        trank = sum(s1 > tol); % t-SVD rank
        %
        U = U(:,1:trank,:);
        U=reshape(U,[n1,trank,Nway(3:end)]);
        V = V(:,1:trank,:);
        V=reshape(V,[n2,trank,Nway(3:end)]);
        S = S(1:trank,1:trank,:);
        S=reshape(S,[trank,trank,Nway(3:end)]);
    end

elseif strcmp(opt,'full') == 1
    U = zeros([n1,n1,Nway(3:end)]);
    S = zeros([n1,n2,Nway(3:end)]);
    V = zeros([n2,n2,Nway(3:end)]);
    [U(:,:,1),S(:,:,1),V(:,:,1)] = svd(X(:,:,1));

    for j = 3 : Ndim
        for i = L(j-1)+1 : L(j)

            I = unfoldi(i,j,L);
            halfnj = floor(Nway(j)/2)+1;
            if I(j) <= halfnj && I(j) >= 2
                [U(:,:,i),S(:,:,i),V(:,:,i)] = svd(X(:,:,i));
                %
            elseif I(j) > halfnj
                n_ = nc(I,j,Nway);
                i_ = foldi(n_,j,L);
                U(:,:,i) = conj(U(:,:,i_));
                V(:,:,i) = conj(V(:,:,i_));
                S(:,:,i) = conj(S(:,:,i_));
            end
        end

    end
end

for jj = Ndim:-1:3
    U = ifft(U,[],jj);
    S = ifft(S,[],jj);
    V = ifft(V,[],jj);
end
 U = real(U);
 S = real(S);
 V = real(V);
end


