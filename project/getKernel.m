function kernel = getKernel()
%% set directory
dir_name = 'best';
titleprefix = strcat(dir_name, '/');
%% get kernel
kernel.num_neuron = 15;
kernel.layer_num = 3;
kernel.layer = cell(kernel.layer_num,1);
kernel.weight = cell(kernel.layer_num,1);
kernel.bias = cell(kernel.layer_num, 1);
kernel.weight{2, 1} = csvread(strcat(titleprefix, 'weight_i2h.csv'));
kernel.weight{3, 1} = csvread(strcat(titleprefix, 'weight_h2o.csv'));
kernel.bias{2, 1} = csvread(strcat(titleprefix, 'bias_i2h.csv'));
kernel.bias{3, 1} = csvread(strcat(titleprefix, 'bias_h2o.csv'));