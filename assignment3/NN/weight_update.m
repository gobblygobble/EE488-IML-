function net = weight_update(net, net_update, l_rate)
%% Your code here %%

num_input = net.num_neuron(1);
num_hidden = net.num_neuron(2);
num_output = net.num_neuron(3);

for h = 1:num_hidden
    for i = 1:num_input
        net.weight{2, 1}(h, i) = net.weight{2, 1}(h, i) - l_rate * net_update.input2hidden(h, i);
    end
end

for o = 1:num_output
    for h = 1:num_hidden
        net.weight{3, 1}(o, h) = net.weight{3, 1}(o, h) - l_rate * net_update.hidden2output(o, h);
    end
end
end