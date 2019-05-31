clear;clc;
%% Your parameter here %%
l_rate = 0.1;
T = 5;
%% Codes %%
[x_train,y_train,x_test,y_test] = createDataset('train.csv','test.csv');
theta = train(x_train,y_train,l_rate,T);
acc = test(x_train,y_train,theta);
% was originally acc = test(x_train,y_train,l_rate,theta) but I changed it
plot_perceptron(x_train,y_train,x_test,y_test,theta);