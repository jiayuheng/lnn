function [a3,accuracy] = lnnpredict(theta, visibleSize, hiddenSize, outputSize, data, Y)

%% the parameters of nn

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize*outputSize), outputSize, hiddenSize);
b1 = theta(hiddenSize*visibleSize+hiddenSize*outputSize+1:hiddenSize*visibleSize+hiddenSize*outputSize+hiddenSize);
b2 = theta(hiddenSize*visibleSize+hiddenSize*outputSize+hiddenSize+1:end);

%% model output and 
[n m] = size(data);


z2 = W1*data+repmat(b1,1,m);
a2 = sigmoid(z2);
z3 = W2*a2+repmat(b2,1,m);
a3 = sigmoid(z3);
[a,index]=max(a3);
clear a 
accuracy=sum((Y==index'))/length(index);

end



function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function sigmInv = sigmoidInv(x)

    sigmInv = sigmoid(x).*(1-sigmoid(x));
end

