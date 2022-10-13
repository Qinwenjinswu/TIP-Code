function bX = H_bcirc(X)
% Input:
%     X     -    Order-d tensor£ºn1*n2*n3*¡­¡­*nd  
% Output:
%    bX    -    block circulant matrix of size (n_1n_3¡­n_d * n_2n_3¡­n_d) 
%
% X_i: order-(d-1) tensor constructed by keeping the d-th index of  order-d tensor X fixed at i,
%  i.e., X_i := X(:,¡­¡­,:,i)
%
%                 [ circ(X_1)     circ( X_{nd} )     circ( X_{nd-1} )     ¡­     circ(X_2); 
%                   circ(X_2)     circ( X_1 )        circ( X_{nd} )       ¡­     circ(X_3); 
%      circ(X)=        ¡­            ¡­                   ¡­              ¡­       ¡­     ;
%                  circ(X_{nd})   circ( X_{nd-1} )   circ( X_{nd-2} )     ¡­     circ(X_1)]
%
% H_bcirc(X): an  (n_1n_3¡­n_d * n_2n_3¡­n_d) block circulant matrix  at the base level of  the operator circ(X),
% in other word, H_bcirc(X) is an (n_1n_3¡­n_d * n_2n_3¡­n_d) matrix formed  by applying circ(X) repeatedly until a block matrix result.
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
Xi=cell(Nway(Ndim),1);
for index = 1 : Nway(Ndim)
    X_permute=permute(X,[Ndim  1:Ndim-1]);
    X_sub=X_permute(index,:);
    Xi{index}=reshape(X_sub,Nway(1:Ndim-1));
    % Xi{index}=get_Ai(X,index);
end

bX_cell=cell(Nway(Ndim),Nway(Ndim));
for i = 0 : Nway(Ndim)-1
    bX_cell(:,i+1) =( circshift(Xi,i));
end

if(ismatrix(bX_cell{1,1}))
    bX=cell2mat(bX_cell);
    return ;
elseif (ndims(bX_cell{1,1})>=2)
    sub_cell=cell(Nway(Ndim),Nway(Ndim));
    for x=1 : Nway(Ndim)
        for y=1 : Nway(Ndim)
            %
            sub_cell{x,y}= H_bcirc(bX_cell{x,y});
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