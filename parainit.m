function theta = parainit(outSize, hiddenSize, visibleSize)


W1 = rand(hiddenSize, visibleSize);
W2 = rand(outSize, hiddenSize);

b1 = zeros(hiddenSize, 1);
b2 = zeros(outSize, 1);


theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

