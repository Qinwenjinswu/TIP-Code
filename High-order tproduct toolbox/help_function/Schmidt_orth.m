function b=Schmidt_orth(a)
%Schmidt_orthogonalization
[m,n] = size(a);
if(m<n)
error('Row less than column, cannot calculate, please transpose and re-enter');
  %return;
end

b=zeros(m,n);
%orthogonalization
b(:,1)=a(:,1);
for i=2:n
for j=1:i-1
b(:,i)=b(:,i)-dot(a(:,i),b(:,j))/dot(b(:,j),b(:,j))*b(:,j);
end
b(:,i)=b(:,i)+a(:,i);
end

%normalization
 for k=1:n
 b(:,k)=b(:,k)/norm(b(:,k));
 end