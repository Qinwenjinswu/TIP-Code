function get_Xi = get_Ai(X,index)

% Written by  Wenjin Qin  (qinwenjin2021@163.com)

Nway=size(X);
Ndim=ndims(X);
Xi_dim=zeros(1,Ndim-1);
for i=1:Ndim-1
    Xi_dim(i)=Nway(i);
end

Xi=zeros(Xi_dim);
X_permute=permute(X,[Ndim  1:Ndim-1]);
X_sub=X_permute(index,:);
get_Xi=Xi+reshape(X_sub,Nway(1:Ndim-1));

