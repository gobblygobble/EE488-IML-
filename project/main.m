%clear; clc;
%% your parameter here%%
param.balance_data = 1; % should we balance the data
param.decrease_learning_rate = 1; % should we decrease learning rate
param.num_epoch = 5; % number of epochs
param.learning_rate = 10.0; % learning rate
param.bias_l_rate = 10.0; % bias learning rate
%% algorithm %%
[x_train,y_train,x_valid,y_valid] = createDataset('train_feat.csv', 'train_label.csv','valid_feat.csv', 'valid_label.csv');

%% training 
%model= algorithm(param, x_train, y_train);
%% MY BEST PERFORMING KERNEL SO FAR
model = getKernel();
%% validation 
[valid_p] = validation(model, x_valid);

%% test 
[x_test, y_test] = createDatasetTest('test_feat.csv', 'test_label.csv');
start_time = clock;
[test_p] = validation(model, x_test);
end_time = clock;
fprintf('CLOCK:  %g\n',etime(start_time, end_time));
%% Find Accuracy
disp('class predict');
disp([y_valid valid_p]);
disp([y_test test_p]);
valid_acc =mean(y_valid== valid_p)*100;
test_acc =mean(y_test == test_p)*100;
fprintf('valid_acc =%d\n',valid_acc);
fprintf('\test_acc =%d\n',test_acc);

%% Save File
% saveTest(model, 'test_feat.csv', 'test_label.csv');