function d = sqdist(a,b)
% sqdist - computes pairwise squared Euclidean distances between points
% 求欧氏距离平方

% original version by Roland Bunschoten, 1999

aa = sum(a.*a,1); bb = sum(b.*b,1);% ab = a'*b; 
%sum（1）行相加
d = - 2*(a'*b);
d = d + repmat(aa',[1 size(bb,2)]);
d = d + repmat(bb,[size(aa,2) 1]);

% d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;
