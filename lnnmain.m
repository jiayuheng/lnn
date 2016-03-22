%this code is based on Stanford University DeepLearning tutorial.
clc
clear

tic
inputSize = 28 * 28;
numClasses = 10;
% set the structure of  of nn
hiddenSize = 100;    % Layer 1 Hidden Size
% hiddenSizeL2 = 64;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average sparse rato.

lambda = 3e-3;         % regularization parameter       
beta = 3;              % sparsity penalty term       


%%  Load data

addpath(genpath('./minFunc_2012/minFunc'));


load MNIST
%% debug
samples=10000;
trainData=trainData(:,1:samples);
trainLabels=trainLabels(1:samples);

trainLabels(trainLabels == 0) = 10; 
yy = full(sparse(trainLabels, 1:samples, 1));
%% Laplacian graph
nnparams=cell(1);
nnparams{1}='knn';
opts.K =4; 
opts.maxblk = 1e7;
opts.metric = 'eucdist';
nnparams{2}=opts;
TG=zeros(10,length(trainLabels));
for i=1:10
    idx=find(trainLabels==i);
    TG(i,idx)=1;
end
   
trainDataTG=[trainData;TG];
T_G=slnngraph(trainDataTG,[],nnparams);
sigmalap=.56;
for i = 1:size(trainDataTG,2)
    ind = find(T_G(:,i)~=0);
    Wlap(ind,i) = exp(-sigmalap*T_G(ind,i));    
end
Wlap = (Wlap+Wlap')*0.5;
Dlap=diag(sum(Wlap,2));
L=Dlap-Wlap;

%% train first sae or can replace this by rbm of something
randomTheta = parainit(numClasses, hiddenSize, inputSize);

% optimization algorithm
options = struct;
options.Method = 'lbfgs';
options.maxIter = 100;
% options.display = 'on';
% [sae1OptTheta, cost] =  minFunc(@(p)saecost(p,inputSize,hiddenSizeL1,lambda,sparsityParam,beta,trainData),sae1Theta,options);% minfunc is a kind of newtown method
% lap= 0.000020;
% lap= 1;
lap= 0;

% lap2=.2;
% lap2=.02;
lap2=.02;
% lap2=.0;

% beta=0;
lambda=3e-4;
% lambda=0;
% sigmalap=1;
% [Wlap,Dlap]=lapWD(trainData,sigmalap);
% L=Dlap-Wlap;
% [sae1OptTheta, cost] =  minFunc(@(p)saecostlap(p,inputSize,hiddenSizeL1,lambda,sparsityParam,beta,trainData,lap,L,lap2),sae1Theta,options);
[theta, cost] =  minFunc(@(p)lnncost(p, inputSize, hiddenSize, numClasses, lambda, lap2 ,trainData, yy, L),randomTheta,options);
%% train softmax
[a3,trainaccuracy] = lnnpredict(theta, inputSize, hiddenSize, numClasses, trainData, trainLabels);
[a3,testaccuracy] = lnnpredict(theta, inputSize, hiddenSize, numClasses, testData, testLabels);
disp(['the train accuracy is: ', num2str(trainaccuracy)]);
disp(['the test accuracy is: ', num2str(testaccuracy)])
% [sae2Features] = saeoutput(sae2OptTheta, hiddenSizeL2, hiddenSizeL1, sae1Features);
% [sae1Features] = saeoutput(sae1OptTheta, hiddenSizeL1, inputSize, trainData);
% 
% 
% saeSoftmaxTheta = 0.5 * randn(hiddenSizeL1 * numClasses, 1);
% 
% softmaxLambda = 1e-4;
% numClasses = 10;
% softoptions = struct;
% softoptions.maxIter = 400;
% softmaxModel = softmaxTrain(hiddenSizeL1,numClasses,softmaxLambda,sae1Features,trainLabels,softoptions);
% saeSoftmaxOptTheta = softmaxModel.optTheta(:);


%% model output

%% fineturn the para by bp


% % stack = cell(2,1);
% stack = cell(1,1);
% stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), hiddenSizeL1, inputSize);
% stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
% 
% % stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1),hiddenSizeL2, hiddenSizeL1);
% % stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);
% 
% % Initialize the parameters for the deep model
% [stackparams, netconfig] = stack2params(stack);%code by andrew ng
% 
% stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];
% lambda = 3e-2
% 
% [stackedAEOptTheta, cost] =  minFunc(@(p)msaecost(p,inputSize,hiddenSizeL1,numClasses, netconfig,lambda, trainData, trainLabels),stackedAETheta,options);
% 
% 
% 
%   optStack = params2stack(stackedAEOptTheta(hiddenSizeL1*numClasses+1:end), netconfig);% ng
%   W11 = optStack{1}.w;
% %   W12 = optStack{2}.w;



%% test


