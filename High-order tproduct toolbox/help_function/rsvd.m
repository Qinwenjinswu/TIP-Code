function [U,Sigma,V] = rsvd(A,k,p,q)
%
% randomized  Singular Value Decomposition (r-SVD) using power iteration 
%
% Input:
%       A       -    n1*n2 matrix
%       k       -   truncation term: k < min(n1,n2)
%       p       -   oversampling parameter: p>0
%       q       -   the number of  power iteration:  1~3 power iterations are sufficient
%
% Output: U,Sigma,V
%
% References:
% Wenjin Qin, Hailin Wang, Weijun Ma, Jianjun Wang. Robust high-order tensor recovery via nonconvex low-rank approximation[C]. 
% In: Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
% 2022: 3633-3637.
%  ResearchGate: https://www.researchgate.net/publication/360423337_Robust_High-Order_Tensor_Recovery_Via_Nonconvex_Low-Rank_Approximation
%
% Written by  Wenjin Qin  (qinwenjin2021@163.com)
%
m = size(A,1);
n = size(A,2);
l = k + p;
R = randn(n,l);

Q = powerMethod( A, R, q);

% [Uhat,Sigmahat,Vhat] = svd(Q'*A,'econ');
% U = Q*Uhat;
% Sigma = Sigmahat;
% V = Vhat;

 Bt = A'*Q;
 [Qhat,Rhat] = qr(Bt,0);
 [Uhat,Sigmahat,Vhat] = svd(Rhat,'econ');
 U = Q*Vhat;
 Sigma = Sigmahat;
 V = Qhat*Uhat;
   
% take first k components
U = U(:,1:k);
Sigma = Sigma(1:k,1:k);
V = V(:,1:k);
end


function Q = powerMethod( A, R, maxIter, tol)

if(~exist('tol', 'var'))
    tol = 1e-5;
end

Y = A*R;
[Q, ~] = qr(Y, 0);
err = zeros(maxIter, 1);
for i = 1:maxIter
%     Y = A*(A'*Q);
%     [iQ, ~] = qr(Y, 0);

   Y1=A'*Q;
   [iQ1, ~] = qr(Y1, 0);
   Y2=A*iQ1;
   [iQ2, ~] = qr(Y2, 0);
   
    err(i) = norm(iQ2(:,1) - Q(:,1), 1);
    Q = iQ2;
    
    if(err(i) < tol)
        break;
    end
end

end

