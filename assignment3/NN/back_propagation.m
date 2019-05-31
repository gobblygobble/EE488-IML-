function net_update = back_propagation(net, data_output)
%% Your code here %%

% current parameters
%input2hidden = net.weight{2, 1};
%hidden2ouput = net.weight{3, 1};
layerI = net.layer{1, 1};
layerH = net.layer{2, 1};
layerO = net.layer{3, 1};
batch_size = size (layerI, 2);

num_input = net.num_neuron(1);
num_hidden = net.num_neuron(2);
num_output = net.num_neuron(3);

net_update.input2hidden = zeros(num_hidden, num_input);
net_update.hidden2output = zeros(num_output, num_hidden);

% matrix introduced to reduce time when computing dL/dw_hi, because of
% repeated components... reduce calculation overhead
saved_par = zeros(num_output, batch_size);

% compute dL/dw_oh for each batch.
% we will add them up, and divide by batch_size to take the average
% target values for 0, and 1 were changed to 0.3 and 0.7, respectively,
% because if put as 0 and 1, weights will grow infinitely
for n = 1:batch_size
    for o = 1:num_output
        saved_par(o, n) = 2 / num_output * (layerO(o, n) - (0.3 + 0.4 * data_output(o, n))) * (layerO(o, n) - layerO(o, n)^2);
        for h = 1:num_hidden
           net_update.hidden2output(o, h) = net_update.hidden2output(o, h) + saved_par(o, n) * layerH(h, n);
        end
    end
end
net_update.hidden2output = net_update.hidden2output / batch_size;

% compute dL/dw_hi for each batch
% we will add them up, and divide by batch_size to take the average
for n = 1:batch_size
    sigma_aid = saved_par(:, n);
    for h = 1:num_hidden
        % sigma = dL/dh_h * dh_h/dh_h(in)
        sigma = sigma_aid' * net.weight{3, 1}(:, h) * (layerH(h, n) - layerH(h, n)^2);
        for i = 1:num_input
            net_update.input2hidden(h, i) = net_update.input2hidden(h, i) + sigma * layerI(i, n);
        end
    end
end
net_update.input2hidden = net_update.input2hidden / batch_size;
end