clear;clc;close all;
%% Your parameter here %%
k = 1;
%% Codes %%
[x_train,y_train,x_test,y_test] = createDataset('train.csv','test.csv');
[class] = classify_knn(x_test,x_train,y_train,k);
[acc] = test_knn(class,y_test);
plot_knn(x_train,y_train,x_test,y_test,k)