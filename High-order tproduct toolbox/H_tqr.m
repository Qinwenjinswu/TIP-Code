function [Q,R] = H_tqr(X,transform,opt)

% Order-d (d>=3) tensor QR-type decomposition under linear transform
%
%   [Q,R] = H_tqr(X,transform), where X is n1*n2*n3*бнбн*nd  tensor, 
%   produces an n1*n2*n3*бнбн*nd f-upper triangular tensor R and an n1*n1*n3*бнбн*nd orthogonal tensor Q so that X = Q*R.
%
%   [Q,R] = H_tqr(X,transform,'econ') produces the "economy size" decomposition.
%   If n1>n2, produces an n2*n2*n3*бнбн*nd f-upper triangular tensor R and an n1*n2*n3*бнбн*nd orthogonal tensor Q so that X = Q*R.
%   If n1<=n2, this is the same as [Q,R] = H_tqr(X,transform).
%
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
%      opt      -   options for different outputs of Q and R:
%                  - 'econ': produces the "economy size" decomposition. 
%                  - If not specified (default), produces full decomposition, i.e., X = Q*R, where
%                     X - n1*n2*n3*бнбн*nd
%                     Q - n1*n1*n3*бнбн*nd
%                     R - n1*n2*n3*бнбн*nd
%
%
% Output: Q, R
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

Ndim = length(size(X));
Nway= size(X);
n1=size(X,1);
n2=size(X,2);
if nargin < 2
    % fft is the default transform
    transform.L = 'fft'; transform.rho = prod(Nway(3:end)); transform.inverseL = 'ifft';
end

if strcmp(transform.L,'fft')
    % efficient computing for fft transform
    L = ones(1,Ndim);
    for x = 3:Ndim
        X = fft(X,[],x);
        L(x) = L(x-1) * Nway(x);
    end
  
    if n1>n2 && exist('opt', 'var') && strcmp(opt,'econ') == 1
        Q = zeros([n1,n2,Nway(3:end)]);
        R = zeros([n2,n2,Nway(3:end)]);
        
        [Q(:,:,1),R(:,:,1)] = qr(X(:,:,1),0);
        
        for j = 3 : Ndim
            for i = L(j-1)+1 : L(j)
                
                I = unfoldi(i,j,L);
                halfnj = floor(Nway(j)/2)+1;
                if I(j) <= halfnj && I(j) >= 2
                    [Q(:,:,i),R(:,:,i)] = qr(X(:,:,i),0);
                elseif I(j) > halfnj
                    n_ = nc(I,j,Nway);
                    i_ = foldi(n_,j,L);
                    Q(:,:,i) = conj(Q(:,:,i_));
                    R(:,:,i) = conj(R(:,:,i_));
                    
                end
            end
        end
        
        
        
    else
        Q = zeros([n1,n1,Nway(3:end)]);
        R = zeros([n1,n2,Nway(3:end)]);
        
        [Q(:,:,1),R(:,:,1)] = qr(X(:,:,1));
        
        for j = 3 : Ndim
            for i = L(j-1)+1 : L(j)
                
                I = unfoldi(i,j,L);
                halfnj = floor(Nway(j)/2)+1;
                if I(j) <= halfnj && I(j) >= 2
                    [Q(:,:,i),R(:,:,i)] = qr(X(:,:,i));
                elseif I(j) > halfnj
                    n_ = nc(I,j,Nway);
                    i_ = foldi(n_,j,L);
                    Q(:,:,i) = conj(Q(:,:,i_));
                    R(:,:,i) = conj(R(:,:,i_));
                    
                end
            end
        end
        
    end
    
    for jj = Ndim:-1:3
        Q = ifft(Q,[],jj);
        R = ifft(R,[],jj);
        
    end
    
    Q= real(Q);
    R = real(R);
    
else
    % other transform
    A = lineartransform(X,transform);
    if n1>n2 && exist('opt', 'var') && strcmp(opt,'econ') == 1
        Q = zeros([n1,n2,Nway(3:end)]);
        R = zeros([n2,n2,Nway(3:end)]);
        for i = 1 : prod(Nway(3:end))
            [Q(:,:,i),R(:,:,i)] = qr(A(:,:,i),0);
        end
    else
        Q = zeros([n1,n1,Nway(3:end)]);
        R = zeros([n1,n2,Nway(3:end)]);
        for i = 1 : prod(Nway(3:end))
            [Q(:,:,i),R(:,:,i)] = qr(A(:,:,i));
        end
    end
    for y = Ndim:-1:3
        Q = tmprod(Q,inv(transform.L{y-2}),y);
        R = tmprod(R,inv(transform.L{y-2}),y);
    end
end
