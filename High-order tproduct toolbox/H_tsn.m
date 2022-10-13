function htsn =H_tsn(A,transform)

%
%
% H_tsn: the tensor spectral norm for the high-order case
% htsn =H_tsn(A,transform) computes the  spectral norm of an order-d (d>=3) tensor under linear transform
%
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
% Output: htsn     -  high-order tensor spectral norm 
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

Nway = size(A);
Ndim= ndims(A);
if nargin < 2
    % fft is the default transform
    transform.L = 'fft'; transform.rho = prod(Nway(3:end)); transform.inverseL = 'ifft';
end

htsn = 0;
if strcmp(transform.L,'fft')
    % efficient computing for fft transform
    L = ones(1,Ndim);
    for y = 3:Ndim
        A = fft(A,[],y);
        L(y) = L(y-1) * Nway(y);
    end
    s = max(svd(A(:,:,1),'econ'));
    htsn = max(htsn,s);
    
    for j = 3 : Ndim
        for i = L(j-1)+1 : L(j)
            I = unfoldi(i,j,L);
            halfnj = floor(Nway(j)/2)+1;
            if I(j) <= halfnj && I(j) >= 2
                s = max(svd(A(:,:,i),'econ'));
               % htnn = max(s,htsn);
            end
        end
    end
    htsn = max(s,htsn);
    
else
    % other transform
    A = lineartransform(A,transform);
    for i = 1 : prod(Nway(3:end))
        s = max(svd(A(:,:,i),'econ'));
        htsn = max(s,htsn);
    end
end
