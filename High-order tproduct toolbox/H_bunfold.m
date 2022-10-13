function bX = H_bunfold(X)

% Input:
%     X     -    Order-d tensor£ºn1*n2*n3*¡­¡­*nd  
% Output:
%    bX    -    block matrix of size (n_1n_3¡­n_d * n_2) 
%
% X_i: order-(d-1) tensor constructed by keeping the d-th index of  order-d tensor X fixed at i,
%  i.e., X_i := X(:,¡­¡­,:,i)
%
%                 [ unfold(X_1)    
%                   unfold(X_2)    
%      unfold(X)=        ¡­        
%                   unfold(X_{nd})]
%
% H_bunfold(X): an  (n_1n_3¡­n_d * n_2) block  matrix  at the base level of  the operator unfold(X),
% in other word, H_bunfold(X) is an (n_1n_3¡­n_d * n_2) matrix formed  by applying unfold(X) repeatedly until a block matrix result.
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
bX_cell=cell(Nway(Ndim),1);
for index = 1 : Nway(Ndim)
  X_permute=permute(X,[Ndim  1:Ndim-1]);
  X_sub=X_permute(index,:);
  bX_cell{index}=reshape(X_sub,Nway(1:Ndim-1));
 % Xi{index}=get_Ai(X,index);
end


if(ismatrix(bX_cell{1,1}))
    bX=cell2mat(bX_cell);
    return ;
elseif (ndims(bX_cell{1,1})>=2)
    sub_cell=cell(Nway(Ndim),1);
    for x=1 : Nway(Ndim)
            %
            sub_cell{x,1}= H_bunfold(bX_cell{x,1});
            %
    end
    %
    if(ismatrix(sub_cell{1,1}))
        bX=cell2mat(sub_cell);
        return;
    end
end
     

end