clear;clc;
%% Your parameter here %%
T = 10000; %the number of iteration
%% Codes %%
[x_train,y_train,x_test,y_test] = createDataset('train.csv','test.csv');
[avg_alpha] = train_rbf(x_train,y_train,T);
acc = test_rbf(x_test,y_test,avg_alpha);
plot_svm_rbf(x_train,y_train,x_test,y_test,avg_alpha);