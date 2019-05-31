function [x_train,y_train,x_test,y_test] = createDataset(train_data,test_data)
%% Arguments %%
% train_data : name of train data
% test_data : name of test data
% x_train : train data
% y_train : train label
% x_test : test data
% y_test : test label
%% Codes %%
train = load(train_data);
test = load(test_data);

x_train = train(:,1:2);
y_train = train(:,3);
x_test = test(:,1:2);
y_test = test(:,3);
end