function I = H_teye(Nway,transform)

%
% I = H_teye(Nway,transform) generates the Identity tensor of size n*n*n3*бнбн*nd under linear transform
%
% Input:
%   Nway=[n,n,n3,бнбн,nd]   -     the size of identity tensor 
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
% Output: 
%        I      -   identity tensor of size n*n*n3*бнбн*nd  
%

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

if Nway (2) ~= Nway(1)
    error('Inner tensor dimensions must agree.');
end

if nargin < 2
    % fft is the default transform
    transform.L = 'fft'; transform.rho = prod(Nway(3:end)); transform.inverseL = 'ifft';
end

I = zeros(Nway);
if strcmp(transform.L,'fft')
    % fft transform
     I(:,:,1) = eye(Nway(1));
else
    % other transform
    for i = 1 : prod(Nway(3:end))
        %I(:,:,i) =eye(Nway(1));
        I(:,:,i) = eye(Nway(1));
    end
    I = inverselineartransform(I,transform);
end

end



