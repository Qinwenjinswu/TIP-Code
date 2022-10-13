function [U ,S ] = H_eigen(X,transform)
%
% Order-d (d>=3) tensor eigenvalue decomposition  under any invertible linear transform
%
% [U,S] = H_eigen(X,transform) computes the order-d tensor eigenvalue decomposition under linear transform, i.e., X=U * S * H_tinv(U), 
%where S is an n*n*n3*бнбн*nd f-diagonal tensor, U is an n*n*n3*бнбн*nd tensor  under linear transform.
%
% Input:
%       X       -   n*n*n3*бнбн*nd tensor
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
% Output: U, S
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
if Nway (2) ~= Nway(1) 
    error('Inner tensor dimensions must agree.');
end
if nargin < 2
    % fft is the default transform
    transform.L = 'fft'; transform.rho = prod(Nway(3:Ndim)); transform.inverseL = 'ifft';
end

if strcmp(transform.L,'fft')
    [U ,S]=H_eigen_fft(X);
else
    X =lineartransform(X,transform);
    
    n1=size(X,1);
    n2=size(X,2);
    U= zeros([n1,n1,Nway(3:end)]);
    S = zeros([n2,n2,Nway(3:end)]);
   
    for i = 1 :prod(Nway(3:end))
        [U(:,:,i),S(:,:,i)] = eig(X(:,:,i));
    end

    U=inverselineartransform(U,transform);
    S=inverselineartransform(S,transform);
end

end


function [U,S] = H_eigen_fft(X)

Ndim = length(size(X));
Nway= size(X);
L = ones(1,Ndim);
for x = 3:Ndim
    X = fft(X,[],x);
    L(x) = L(x-1) * Nway(x);
end

n1=size(X,1);
n2=size(X,2);
U= zeros([n1,n1,Nway(3:end)]);
S = zeros([n2,n2,Nway(3:end)]);

[U(:,:,1),S(:,:,1)] = eig(X(:,:,1));

for j = 3 : Ndim
    for i = L(j-1)+1 : L(j)
        
        I = unfoldi(i,j,L);
        halfnj = floor(Nway(j)/2)+1;
        if I(j) <= halfnj && I(j) >= 2
            [U(:,:,i),S(:,:,i)] = eig(X(:,:,i));
        elseif I(j) > halfnj
            n_ = nc(I,j,Nway);
            i_ = foldi(n_,j,L);
            U(:,:,i) = conj(U(:,:,i_));
            %
            S(:,:,i) = conj(S(:,:,i_));
        end
        
    end
end

for jj = Ndim:-1:3
    U = ifft(U,[],jj);
    S = ifft(S,[],jj);
end
 U = real(U);
 S = real(S);
end


