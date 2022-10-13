function trank = H_trank(X,transform,tol)

%
% trank = H_trank(X,transform,tol) computes the t-SVD rank of an order-d(d>=3) tensor under linear transform
%
% Input:
%       X      -   Order-d tensor£ºn1*n2*n3*¡­¡­*nd    
%   transform   -   a structure which defines the linear transform
%       transform.L: the linear transform of two types:
%                  - type I: function handle, i.e., @fft, @dct
%                  - type II: invertible  transform  matrix of size ni*ni, for i=3,¡­¡­,d
%
%       transform.inverseL: the inverse linear transform of transform.L
%                         - type I: function handle, i.e., @ifft, @idct
%                         - type II: inverse  transform  matrix of transform.L
%
%       transform.rho: a constant which indicates whether the following property holds for the linear transform or not:
%                    U_3'*U_3=U_3*U_3'=l3*I,¡­, U_d'*U_d=U_d*U_d'=ld*I, for some l3>0,¡­,ld>0
%                    let rho=l3*¡­*ld
%                  - transform.rho > 0: indicates that the above property holds. Then we set transform.rho = rho.
%                  - transform.rho < 0: indicates that the above property does not hold. Then we can set transform.rho = c, for any constant c < 0.
%
%       If not specified, fft is the default transform, i.e.,
%       transform.L = @fft, transform.rho = n3*¡­¡­*nd, transform.inverseL = @ifft. 
%
%
%
% Output: trank    -  the t-SVD rank of an order-d(d>=3) tensor   
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

Nway=size(X);
Ndim=ndims(X);
if nargin < 2
    % fft is the default transform
    transform.L = 'fft'; transform.rho = prod(Nway(3:end)); transform.inverseL = 'ifft';
end

n1=size(X,1);
n2=size(X,2);
min12=min(n1,n2);
s=zeros(min12,1);
S = zeros([min12,min12,Nway(3:end)]);

if strcmp(transform.L,'fft')
    L = ones(1,Ndim);
    for i = 3:Ndim
        X = fft(X,[],i);
        L(i) = L(i-1) * Nway(i);
    end
    
    [~,S(:,:,1),~] = svd(X(:,:,1),'econ');
    s = s+diag(S(:,:,1));
      
    for j = 3 : Ndim
        for i = L(j-1)+1 : L(j)
            I = unfoldi(i,j,L);
            %halfnj = round(Nway(j)/2);
            halfnj = floor(Nway(j)/2)+1;
            if I(j) <= halfnj && I(j) >= 2
                [~,S(:,:,i),~] = svd(X(:,:,i),'econ');
                s = s+diag(S(:,:,i));
            elseif I(j) > halfnj
                n_ = nc(I,j,Nway);
                i_ = foldi(n_,j,L);
                S(:,:,i) = conj(S(:,:,i_));
                s = s+diag( S(:,:,i) );
            end
        end
    end
    
    s = s/prod(Nway(3:end))^2;
    if nargin < 3
        tol = max(Nway(1),Nway(2)) * eps(max(s));
    end
    trank = sum(s > tol);
    
else
    % other transform
    X = lineartransform(X,transform);
    for i=1:prod(Nway(3:end))
        s = s + svd(X(:,:,i),'econ');
    end
    
    if transform.rho > 0
        % property U_3'*U_3=U_3*U_3'=l3*I ,¡­, U_d'*U_d=U_d*U_d'=ld*I, for some l3>0,¡­,ld>0 holds
        % let rho=l3*¡­*ld
        s = s/prod(Nway(3:end))/transform.rho;
    else
        % the above property does not hold for transform L
        transformrho=0;
        for x=3:Ndim
           transformrho=transformrho+norm(transform.L{x-2})^2;
        end
        s = s/prod(Nway(3:end))/transformrho;
    end
    
    if nargin < 3
        tol = 1e-10;
    end
    trank = sum(s > tol);
end

