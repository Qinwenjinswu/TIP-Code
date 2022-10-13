function C = H_tprod(A,B,transform)

%
% C = H_tprod(A,B,transform) computes the tensor-tensor product of two order-d (d>=3)  tensors A and B under linear transform
%
% Input:
%       A       -   Order-d tensor£ºn1*n2*n3*¡­¡­*nd
%       B       -   Order-d tensor: n2*l*n3*¡­¡­*nd
%   transform   -   a structure which defines the linear transform
%       transform.L: the linear transform of two types:
%                  - type I: function handle, i.e., @fft, @dct
%                  - type II: invertible  transform  matrix of size ni*ni, for i=3,¡­¡­,d
%
%       transform.inverseL: the inverse linear transform of transform.L
%                         - type I: function handle, i.e., @ifft, @idct
%                         - type II: inverse transform  matrix of transform.L
%
%       transform.rho: a constant which indicates whether the following property holds for the linear transform or not:
%                    U_3'*U_3=U_3*U_3'=l3*I,¡­, U_d'*U_d=U_d*U_d'=ld*I, for some l3>0,¡­,ld>0
%                    let rho=l3*¡­*ld
%                  - transform.rho > 0: indicates that the above property holds. Then we set transform.rho = rho.
%                  - transform.rho < 0: indicates that the above property does not hold. Then we can set transform.rho = c, for any constant c < 0.
%       If not specified, fft is the default transform, i.e.,
%       transform.L = @fft, transform.rho = n3*¡­¡­*nd, transform.inverseL = @ifft. 
%
%
% Output: C     -   Order-d tensor: n1*l*n3*¡­¡­*nd , C = A * B 
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

NwayA = size(A);
NdimA= ndims(A);
NwayB = size(B);
NdimB= ndims(B);
for x = 3:NdimB
    if NwayA (x) ~= NwayB(x)
        t = 1;
        break;
    else
        t = 0;
    end
end
%
if NwayA (2) ~= NwayB(1) || t == 1
    error('Inner tensor dimensions must agree.');
end

%
if nargin < 3
    % fft is the default transform
    transform.L = 'fft'; transform.rho = prod(NwayA(3:end)); transform.inverseL = 'ifft';
end

%
C = zeros([ size(A,1)  size(B,2)  NwayA(3:end) ]);

if strcmp(transform.L,'fft')
    C=H_tprod_fft(A,B);
elseif iscell(transform.L)
    
    for y = 3:NdimA
        A=tmprod(A,transform.L{y-2},y);
        B=tmprod(B,transform.L{y-2},y);
    end
    
    for z = 1:prod(NwayA(3:NdimA))
        C(:,:,z)=A(:,:,z)*B(:,:,z);
    end
    
    for v = NdimA:-1:3
        C = tmprod(C,inv(transform.L{v-2}),v);
    end
    
end

end




function C = H_tprod_fft(A,B)

% Tensor-tensor product of two order-d A and B tensors under FFT
%
% A - Order-p tensor£ºn1*n2*n3*¡­¡­*np
% B - Order-p tensor: n2*l*n3*¡­¡­*np
% C - Order-p tensor: n1*l*n3*¡­¡­*np
%
% Written by  Wenjin Qin  (qinwenjin2021@163.com)


NwayA = size(A);
NdimA= ndims(A);
NwayB = size(B);
NdimB= ndims(B);
for x = 3:NdimA
    if NwayB(x) ~= NwayA(x)
        t = 1;
        break;
    else
        t = 0;
    end
end
%
if NwayA(2) ~= NwayB(1) || t == 1
    error('Inner tensor dimensions must agree.');
end

C = zeros([ size(A,1)  size(B,2)  NwayA(3:end) ]);
L = ones(1,NdimB);
for y = 3:NdimB
    A = fft(A,[],y);
    B = fft(B,[],y);
    L(y) = L(y-1) * NwayA(y);
end

C(:,:,1) = A(:,:,1)*B(:,:,1);   
for j = 3 : NdimA
    for i = L(j-1)+1 : L(j)
        I = unfoldi(i,j,L);
        halfnj = floor(NwayA(j)/2)+1;
        if I(j) <= halfnj && I(j) >= 2
            C(:,:,i) = A(:,:,i)*B(:,:,i);   
        elseif I(j) > halfnj
            n_ = nc(I,j,NwayA);
            i_ = foldi(n_,j,L);
            C(:,:,i) = conj(C(:,:,i_));
        end
    end
end

for z = NdimA:-1:3
    C = (ifft(C,[],z));
end
C = real(C);
end




