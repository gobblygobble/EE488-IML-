clear all; clc;close all
%% Load data
[x_train,y_train,x_test,y_test] = createDataset();

%% data & neuron number setting
input_dim = size(x_train,2);
num_data_train = size(x_train, 1);
num_data_test = size(x_test, 1);
num_neuron_input = size(x_test, 2);
num_neuron_output = size(y_test, 2);

%% Your parameters here %%
batch_size=4;
num_neuron_hidden = [15]; % hidden neuron num
% weight initialization setting
init.weight_std = 0.1; % stdev of weight paramters
init.bias_std = 0.1; % stdev of bias paramters
% training setting
training.num_epoch = 5; % num of epochs
training.learning_rate = 0.05; % learning rate
training.test_period = 10000; % test every iteration

%% Initializations %%
training.current_epoch = 0;
num_neuron = [num_neuron_input; num_neuron_hidden; num_neuron_output]; 
training.num_input_graph = training.num_epoch * floor(num_data_train / training.test_period);
training.graph = zeros(training.num_epoch, 1);
training.index_graph = 0;

%% initialize network %%
net = initialize_network(num_neuron, init);
k = 0;

%% Training %%
for epoch = 1 : training.num_epoch
      training.current_epoch = epoch;
      order_index_train = randperm(num_data_train);
      for index_data = 1 :batch_size: num_data_train
            data_input = x_train(order_index_train(index_data:index_data+batch_size-1), :)';
            data_output = y_train(order_index_train(index_data:index_data+batch_size-1), :)';
            %% Foward computations
            [net,~] = feed_foward(data_input, net);
            %% Backward computations
            net_update = back_propagation(net, data_output);
            %% Weight update
            net = weight_update(net, net_update, training.learning_rate);
            k = k+1;
            % test step
            if mod(k, training.test_period) == 0 % Test period
                  test_error = 0;
                  training.index_graph = training.index_graph + 1;
                  for index_data_test = 1 : num_data_test
                        data_input = x_test(index_data_test, :)';
                        data_output = y_test(index_data_test, :)';
                        [net, pred] = feed_foward(data_input, net);
                        [~,ind] = max(pred);
                        [~,ind_gt] = max(data_output);
                        if ind_gt ~= ind
                              test_error = test_error + 1;
                        end
                  end
                  test_error = 1 - test_error / num_data_test;
                  training.graph(training.index_graph) = 1 - test_error;
                  % command text
                  performance_test = ['epoch: ', num2str(epoch), '  iteration: ', num2str(k), '  test acc: ', num2str(test_error)];
                  disp(performance_test);
            end 
      end
end

%% Plot graph
plot(training.graph, 'r*-')
title_text = ['Performance - epoch: ', num2str(epoch), ' / Test per epoch: ', num2str(floor(num_data_train / training.test_period))];
title(title_text);




