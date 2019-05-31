%% your classifer traing code here
function [model] = algorithm(param, xx_train, yy_train)
%% training settings - subject to change
balance_data = param.balance_data;
decrease_learning_rate = param.decrease_learning_rate;
num_epoch = param.num_epoch;
learning_rate = param.learning_rate;
bias_l_rate = param.bias_l_rate;
%% Matrix settings
if balance_data == 1
    % balance data
    x_train = xx_train';
    [x_train, yy_train] = remedy_imbalanced(x_train, yy_train);
    y_train = zeros(size(yy_train, 1), 10);
    for i = 1 : size(y_train, 1)
        y_train(i, yy_train(i, 1)) = 1;
    end
else
    % do NOT balance data -> use imbalanced data
    x_train = xx_train';
    y_train = zeros(size(yy_train, 1), 10);
    for i = 1 : size(y_train, 1)
        y_train(i, yy_train(i, 1)) = 1;
    end
end
%% Parameter settings
batch_size = 4;
num_data_train = size(x_train, 1);
num_neuron_input = size(x_train, 2); % = 128
num_neuron_hidden = 15;
num_neuron_output = size(y_train, 2); % = 10
% weight iniialization setting
init.weight_std = 0.1; % stdev of weight parameters
init.bias_std = 0.1; % stdev of bias parameters
% training setting
num_neuron = [num_neuron_input; num_neuron_hidden; num_neuron_output];
net = initialize_network(num_neuron, init);
%% Training
for epoch = 1 : num_epoch
    if decrease_learning_rate == 1
        slowing_factor = epoch;
    else
        slowing_factor = 1;
    end
    order_index_train = randperm(num_data_train);
    for index_data = 1 : batch_size : num_data_train
        data_input = x_train(order_index_train(index_data:index_data+batch_size-1), :)';
        data_output = y_train(order_index_train(index_data:index_data+batch_size-1), :)';
        %% Foward computations
        [net,~] = feed_foward(data_input, net);
        %% Backward computations
        net_update = backpropagation(net, data_output);
        %% Weight update
        net = weight_update(net, net_update, learning_rate / slowing_factor, bias_l_rate / slowing_factor);
    end
    % test - for plotting performance vs epoch
    %x_valid = csvread('valid_feat.csv');
    %y_valid = csvread('valid_label.csv');
    %[valid_p] = validation(net, x_valid);
    %valid_acc = mean(y_valid== valid_p)*100;
    %fprintf('epoch #%d: validation data accuracy =%d\n',epoch, valid_acc);
end
model = net;
end