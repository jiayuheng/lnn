function [Wlap,Dlap]=lapWD(V,sigma)
d=sqdist(V,V);
d=d./max(max(d));
Wlap=zeros(size(V,2),size(V,2));
Dlap=zeros(size(V,2),size(V,2));
lengthlap=length(V);

%old version
% for i=1:length(Wlap)
%     for j=1:length(Dlap)
%         Wlap(i,j)=exp(norm(V(:,i)-V(:,j))/(-1*sigma*sigma));
%     end
% end

%faster version
% for i=1:length(Walp)
%     temp=repmat(Walp(:,j),1,lengthlap);
%     temp=temp-V;
%     Wlap(i,:)=exp(norm(temp(:,:))/(-1*sigma*sigma));
% end


Wlap=exp(d/(-1*sigma*sigma));
% Wlap=1./d;

Dlap=diag(sum(Wlap,2));

