function net_update = backpropagation(net, data_output)
%% current parameters
layerI = net.layer{1, 1};
layerH = net.layer{2, 1};
layerO = net.layer{3, 1};
batch_size = size (layerI, 2);

num_input = net.num_neuron(1);
num_hidden = net.num_neuron(2);
num_output = net.num_neuron(3);

net_update.input2hidden = zeros(num_hidden, num_input);
net_update.hidden2output = zeros(num_output, num_hidden);
% bias update values
net_update.bias_i2h = zeros(num_hidden, 1);
net_update.bias_h2o = zeros(num_output, 1);

% matrix introduced to reduce time when computing dL/dw_hi, because of
% repeated components... reduce calculation overhead
saved_par = zeros(num_output, batch_size);
% compute dL/dw_oh for each batch.
% we will add them up, and divide by batch_size to take the average
% target values for 0, and 1 were changed to 0.3 and 0.7, respectively,
% because if put as 0 and 1, weights will grow infinitely
for n = 1 : batch_size
    for o = 1 : num_output
        saved_par(o, n) = 2 / num_output * (layerO(o, n) - (0.3 + 0.4 * data_output(o, n))) * (layerO(o, n) - layerO(o, n)^2);
        net_update.bias_h2o(o, 1) = net_update.bias_h2o(o, 1) + saved_par(o, n);
        %saved_par(o, n) = 2 / num_output * (layerO(o, n) - (0.2 + 0.6 * data_output(o, n))) * (layerO(o, n) - layerO(o, n)^2);
        for h = 1 : num_hidden
           net_update.hidden2output(o, h) = net_update.hidden2output(o, h) + saved_par(o, n) * layerH(h, n);
        end
        % save bias update value
    end
end
net_update.hidden2output = net_update.hidden2output / batch_size;
net_update.bias_h2o = net_update.bias_h2o / batch_size;

% compute dL/dw_hi for each batch
% we will add them up, and divide by batch_size to take the average
for n = 1 : batch_size
    sigma_aid = saved_par(:, n);
    for h = 1 : num_hidden
        % sigma = dL/dh_h * dh_h/dh_h(in)
        sigma = sigma_aid' * net.weight{3, 1}(:, h) * (layerH(h, n) - layerH(h, n)^2);
        net_update.bias_i2h(h, 1) = net_update.bias_i2h(h, 1) + sigma;
        for i = 1 : num_input
            net_update.input2hidden(h, i) = net_update.input2hidden(h, i) + sigma * layerI(i, n);
        end
    end
end
net_update.input2hidden = net_update.input2hidden / batch_size;
net_update.bias_i2h = net_update.bias_i2h / batch_size;
end