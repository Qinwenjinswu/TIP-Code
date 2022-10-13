function bX = H_b_hankel(X)

% Input:
%     X     -    Order-d tensor£ºn1*n2*n3*¡­¡­*nd  
% Output:
%    bX    -     block hankel matrix of size (n_1n_3¡­n_d * n_2n_3¡­n_d) 
%
% X_i: order-(d-1) tensor constructed by keeping the d-th index of  order-d tensor X fixed at i,
%  i.e., X_i := X(:,¡­¡­,:,i)
%
%                 [ hankel(X_2)          hankel(X_3)       ¡­      hankel( X_{nd} )      0; 
%                   hankel(X_3)          hankel(X_4)       ¡­            0           hankel(X_{nd}); 
%      hankel(X)=      ¡­                   ¡­             ¡­           ¡­              ¡­        ;
%                  hankel(X_{nd})           0              ¡­       hankel(X_4)       hankel(X_3) ;
%                      0               hankel( X_{nd} )    ¡­      hankel(X_3)       hankel(X_2) ]
%
% H_b_hankel(X): an  (n_1n_3¡­n_d * n_2n_3¡­n_d) block hankel matrix  at the base level of  the operator hankel(X),
% in other word, H_b_hankel(X) is an (n_1n_3¡­n_d * n_2n_3¡­n_d) matrix formed  by applying hankel(X) repeatedly until a block matrix result.
%
%
%
% References:
% Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University. 
% June, 2018. https://github.com/canyilu/tproduct.
%
% Canyi Lu, Tensor-Tensor Product Toolbox 2.0. Carnegie Mellon University. 
% April, 2021. https://github.com/canyilu/Tensor-tensor-product-toolbox.
%
% Wen-Hao Xu,  Xi-Le Zhao,  Michael Ng. A fast algorithm for cosine transform based tensor singular value decomposition.
% 2019, arXiv:1902.03070.
%
% Carla D  Martin, Richard Shafer, and Betsy LaRue.
% An order-p tensor factorization with applications in imaging[J]. 
% SIAM Journal on Scientific Computing, 2013, 35(1):  A474 - A490.
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
Ai_Index=toeplitz([2:Nway(Ndim) 0]);
x=1:Nway(Ndim);
y=Nway(Ndim):-1:1;
for z=1:length(x)
    Ai_Index(x(z),y(z))=0;
end

bX_cell=cell(Nway(Ndim),Nway(Ndim));
Xi_dim=zeros(1,Ndim-1);
for i=1:Ndim-1
    Xi_dim(i)=Nway(i);
end

for i = 1 : Nway(Ndim)
    for j=1:Nway(Ndim)
        if Ai_Index(i,j)==0
            bX_cell{i,j}=zeros(Xi_dim);
        else
            bX_cell{i,j}=get_Ai(X,Ai_Index(i,j)); 
        end
    end
end

if(ismatrix(bX_cell{1,1}))
    bX=cell2mat(bX_cell);
    return ;
elseif (ndims(bX_cell{1,1})>=2)
    sub_cell=cell(Nway(Ndim),Nway(Ndim));
    for x=1 : Nway(Ndim)
        for y=1 : Nway(Ndim)
            %
            sub_cell{x,y}= H_b_hankel(bX_cell{x,y});
            %
        end
    end
    %
    if(ismatrix(sub_cell{1,1}))
        bX=cell2mat(sub_cell);
        return;
    end
end
end