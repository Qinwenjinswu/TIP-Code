function A_inv = H_tinv(A,transform)

%
% A_inv = H_tinv(A,transform) computes the inverse of an order-d (d>=3) tensor under linear transform
% 
% Input:
%       A       -   Order-d tensor£ºn1*n2*n3*¡­¡­*nd    
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
% Output: A_inv      -   n1*n2*n3*¡­¡­*nd tensor
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


Ndim=ndims(A);
Nway=size(A);
A_inv=zeros(Nway);

if Nway (2) ~= Nway(1) 
    error('Inner tensor dimensions must agree.');
end

if nargin < 2
    % fft is the default transform
    transform.L = 'fft'; transform.rho = prod(Nway(3:end)); transform.inverseL = 'ifft';
end

I = eye( size(A,1) );
if strcmp(transform.L,'fft')
    
    % efficient computing for fft transform
    L = ones(1,Ndim);
    for y = 3:Ndim
        A = fft(A,[],y);
        L(y) = L(y-1) * Nway(y);
    end
    
    %A_inv(:,:,1) = A(:,:,1)\I;
    A_inv(:,:,1) = pinv(A(:,:,1));
    
    for j = 3 : Ndim
        for i = L(j-1)+1 : L(j)
            I = unfoldi(i,j,L);
            halfnj = floor(Nway(j)/2)+1;
            if I(j) <= halfnj && I(j) >= 2
                %A_inv(:,:,i) = A(:,:,i)\I;
                %A_inv(:,:,i) = inv(A(:,:,i));
                A_inv(:,:,i) = pinv(A(:,:,i)); 
            elseif I(j) > halfnj
                n_ = nc(I,j,Nway);
                i_ = foldi(n_,j,L);
                A_inv(:,:,i) = conj(A_inv(:,:,i_));
            end
        end
    end
    
    for z = Ndim:-1:3
        A_inv = (ifft(A_inv,[],z));
    end
    
    
else
    % other transform
    A = lineartransform(A,transform);
    for i = 1 : prod(Nway(3:end))
        %A_inv(:,:,i) = A(:,:,i)\I;
        A_inv(:,:,i) = pinv(A(:,:,i));
    end
    A_inv = inverselineartransform(A_inv,transform);
end
