function EYA = H_appro(A,k,transform,opt)

% Optimal k-term approximation for higher-order tensors (also called low-rank high-order tensor approximation)
%
% Input:
%       k       -    k < min(n1,n2)
%       A       -   n1*n2*n3*бнбн*nd tensor
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
%  opt  -  the scheme of t-SVD decomposition:  'econ', 'full', 'skinny'
%
% Output: EYA  - the result of optimal k-term approximation
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


NwayA= size(A);
NdimA= length(NwayA);
n1=size(A,1);
n2=size(A,2);
%EYA=zeros(NwayA);
if nargin < 4
    opt = 'econ';
end

if nargin < 3
    % fft is the default transform
    transform.L = 'fft'; transform.l = prod(NwayA(3:NdimA)); transform.inverseL = 'ifft';
end

if nargin < 2
    k = H_trank(A,transform);
end


[U,S,V]=H_tsvd(A,transform,opt);

U = U(:,1:k,:);
U=reshape(U,[n1,k,NwayA(3:NdimA)]);
V = V(:,1:k,:);
V=reshape(V,[n2,k,NwayA(3:NdimA)]);
S = S(1:k,1:k,:);
S=reshape(S,[k,k,NwayA(3:NdimA)]);

EYA= H_tprod( H_tprod(U,S,transform),H_tran(V,transform),transform);



