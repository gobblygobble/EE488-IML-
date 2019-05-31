clear;clc;
%% Your parameter here %%
T = 1000;%the number of iteration
%% Codes %%
[x_train,y_train,x_test,y_test] = createDataset('train.csv','test.csv');
theta = train(x_train,y_train,T);
acc = test(x_train, y_train, theta);
% was originally acc = test(x_train,y_train,eta,theta);
plot_svm(x_train,y_train,x_test,y_test,theta);