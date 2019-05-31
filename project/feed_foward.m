function [net, pred] = feed_foward(input, net)

%% Your code here %%
% activation function: sigmoid
function result = activation_function(x)
    result = 1./(1 + exp(-x));
end
net.layer{1,1} = input;
for index_layer = 2 : net.layer_num
      net.layer{index_layer, 1} = net.weight{index_layer, 1} * net.layer{index_layer-1, 1} + net.bias{index_layer, 1};
      net.layer{index_layer, 1} = activation_function(net.layer{index_layer, 1});
end

[~,ind] = max(net.layer{index_layer, 1});
pred = zeros(size(net.layer{index_layer, 1}));
for i=1 : size(ind,2)
    pred(ind(i),i) = 1;
end
end

