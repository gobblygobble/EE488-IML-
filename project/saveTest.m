%% SAVES TEST VALUE
function [foo]  = saveTest(model, read_file, dirname)
mkdir(dirname);
titleprefix = strcat(dirname, '/');
x_test = csvread(read_file);
[test_p] = validation(model, x_test);
saveMatrix = zeros(4210, 1);
for row = 2 : 4210
    saveMatrix(row, 1) = row - 1;
end
saveMatrix(2:4210, 2) = test_p;
csvwrite(strcat(titleprefix, 'test_label.csv'), saveMatrix);
csvwrite(strcat(titleprefix, 'weight_i2h.csv'), model.weight{2, 1});
csvwrite(strcat(titleprefix, 'weight_h2o.csv'), model.weight{3, 1});
csvwrite(strcat(titleprefix, 'bias_i2h.csv'), model.bias{2, 1});
csvwrite(strcat(titleprefix, 'bias_h2o.csv'), model.bias{3, 1});
end