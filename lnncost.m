function [cost,grad] = lnncost(theta, visibleSize, hiddenSize,outputSize, lambda, lap2 ,data, Y, L)

%% the parameters of nn

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize*outputSize), outputSize, hiddenSize);
b1 = theta(hiddenSize*visibleSize+hiddenSize*outputSize+1:hiddenSize*visibleSize+hiddenSize*outputSize+hiddenSize);
b2 = theta(hiddenSize*visibleSize+hiddenSize*outputSize+hiddenSize+1:end);

%% Cost and gradient variables 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

% cal the grad and cost

Jcost = 0;
Jweight = 0;
% Jsparse = 0;
Jlap = 0;

[n m] = size(data);


z2 = W1*data+repmat(b1,1,m);
a2 = sigmoid(z2);
z3 = W2*a2+repmat(b2,1,m);
a3 = sigmoid(z3);
% size(z2);%200*1000: num of hidden * num of samples
% V=z2;
V2=z3;
% sigmalap=1;
% [Wlap,Dlap]=lapWD(V,sigmalap);
% L=Dlap-Wlap;
% Jlap=lap*trace(V*L*V')*(0.5/(m));

Jlap2=lap2*trace(V2*L*V2')*(0.5/(m));


Jcost = (0.5/m)*sum(sum((a3-Y).^2));


Jweight = (1/2)*(sum(sum(W1.^2))+sum(sum(W2.^2)));


% rho = (1/m).*sum(a2,2);
% Jsparse = sum(sparsityParam.*log(sparsityParam./rho)+ (1-sparsityParam).*log((1-sparsityParam)./(1-rho)));


% cost = Jcost+lambda*Jweight+beta*Jsparse+lap*Jlap+lap2*Jlap2;
cost = Jcost+lambda*Jweight+lap2*Jlap2;

%% bp part 
d3 = -(Y-a3).*sigmoidInv(z3)+lap2*V2*(L+L');
% sterm = beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));
% d2 = (W2'*d3+repmat(sterm,1,m)).*sigmoidInv(z2); 
d2 = (W2'*d3).*sigmoidInv(z2); 



W1grad = W1grad+d2*data';
W1grad = (1/m)*W1grad+lambda*W1;
% size(W1grad)
% for lap part
% W1grad = W1grad+lap*(V*(L+L')*data')/(m);

  
W2grad = W2grad+d3*a2';
W2grad = (1/m).*W2grad+lambda*W2;


b1grad = b1grad+sum(d2,2);
b1grad = (1/m)*b1grad;


b2grad = b2grad+sum(d3,2);
b2grad = (1/m)*b2grad;
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end



function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function sigmInv = sigmoidInv(x)

    sigmInv = sigmoid(x).*(1-sigmoid(x));
end

