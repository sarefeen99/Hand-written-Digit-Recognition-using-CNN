close all
clearvars
clc

% The MNIST image extraction code is taken from
% http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
images = images';
% table = [images labels];
% csvwrite('table.dat',table);
% ttds = tabularTextDatastore('table.dat');

%% Storing all the extraced images into a 3D array.

for i = 1:60000   
    I(:,:,1,i) = reshape(images(i,:),[28,28]);
end
clear images

%% Showing some of the random images form the stored dataset

figure; 
perm = randperm(60000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(I(:,:,1,perm(i)));
end

%% counting the number of examples for each Labels and defining size of image

for i=1:10
    lcount(i,1) = length(find(labels==(i-1)));
end

image_size = size(I(:,:,1));
image_size = [image_size 1]; % 1 = if one channer, 3 = if rgb channels
Total_N = length(I);

%% Splitting Training Data set into Validation set and Training Set

p_train = 0.95;
p_valid = 1-p_train;

n_train = floor(Total_N*p_train);
n_valid = floor(Total_N*p_valid);

index_valid = randperm(Total_N,n_valid);

valid_set = I(:,:,1,index_valid);
valid_labels = categorical(labels(index_valid));

train_set=I;
train_labels=categorical(labels);

train_set(:,:,:,index_valid)=[]; 
train_labels(index_valid)=[];

clear labels I 

%% Converting to matrix for matlab neural network founction

train_cell = {train_set, train_labels};
valid_cell = {valid_set, valid_labels};


%% Setting up the NN Layers

layers = [
    imageInputLayer(image_size)
    
    convolution2dLayer(4,3,'Stride',[4 4],'Padding','same',...
    'WeightLearnRateFactor',1,'BiasLearnRateFactor',1,...
    'WeightL2Factor',1,'BiasL2Factor',1,'Name','A')
    
    batchNormalizationLayer('Name','A1') % creates default batch normalization layer
    
    reluLayer('Name','A2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','A3')
    
    convolution2dLayer(4,3,'Stride',[4 4],'Padding','same',...
    'WeightLearnRateFactor',1,'BiasLearnRateFactor',1,...
    'WeightL2Factor',1,'BiasL2Factor',1,'Name','B')
    
    batchNormalizationLayer('Name','B1')
    
    reluLayer('Name','B2')
     
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Manually initialize the weights from a Gaussian distribution with standard deviation of 0.0001.
% 
% layer.Weights = randn([4 4 1 3]) * 0.0001;
% % Initialize the biases from a Gaussian distribution with a mean of 1 and a standard deviation of 0.00001.
% layer.Bias = randn([1 1 32])*0.00001 + 1;


analyzeNetwork(layers)


%% Specify Training Options and Run training network

options = trainingOptions('sgdm','MaxEpochs',4, 'ValidationData',valid_cell, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(train_set,train_labels,layers,options);

%% Classify Validation Images and Compute Accuracy

YPred = classify(net,valid_set);

accuracy = sum(YPred == valid_labels)/numel(valid_labels)