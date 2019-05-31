function [x_train,y_train,x_test,y_test] = createDataset()
%% Codes %%
addpath ./data;
x_train = loadMNISTImages('./data/train-images-idx3-ubyte')';
y_train = loadMNISTLabels('./data/train-labels-idx1-ubyte');
y_train(y_train==0) = 10; % Remap 0 to 10
tt = zeros(size(y_train,1),10);
tt(sub2ind(size(tt),1:size(y_train,1),y_train'))=1;
y_train = tt;
%load test data
x_test = loadMNISTImages('./data/t10k-images-idx3-ubyte')';
y_test = loadMNISTLabels('./data/t10k-labels-idx1-ubyte');
y_test(y_test==0) = 10; % Remap 0 to 10
tt = zeros(size(y_test,1),10);
tt(sub2ind(size(tt),1:size(y_test,1),y_test'))=1;
y_test = tt;
end