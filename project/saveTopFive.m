%% Runs ITERATIONS iterations of tests and saves the top 5 values
function saveCell = saveTopFive(iterations)
[x_train,y_train,x_valid,y_valid] = createDataset('train_feat.csv', 'train_label.csv','valid_feat.csv', 'valid_label.csv');
saveCell = cell(5, 2);
maxAccuracy = 0;
rowToSave = 1;
savedNum = 0;
% paramaeters
param.balance_data = 0; % should we balance the data
param.decrease_learning_rate = 1; % should we decrease learning rate
param.num_epoch = 5; % number of epochs
param.learning_rate = 10.0; % learning rate
param.bias_l_rate = 10.0; % bias learning rate
for it = 1 : iterations
    model= algorithm(param, x_train, y_train);
    [valid_p] = validation(model, x_valid);
    valid_acc =mean(y_valid== valid_p)*100;
    if valid_acc > maxAccuracy
        % save data
        fprintf('Overwriting accuracy %d with accuracy %d\n', saveCell{rowToSave, 1}, valid_acc);
        savedNum = savedNum + 1;
        maxAccuracy = valid_acc;
        saveCell{rowToSave, 1} = maxAccuracy;
        saveCell{rowToSave, 2} = model;
        if rowToSave == 5
            rowToSave = 1;
        else
            rowToSave = rowToSave + 1;
        end
    end
    if mod(it, 10) == 0
        fprintf('Finished %d iterations - saved %d so far.\n', it, savedNum);
    end
end

%% save 5 files
saveTest(saveCell{1, 2}, 'test_feat.csv', '1');
saveTest(saveCell{2, 2}, 'test_feat.csv', '2');
saveTest(saveCell{3, 2}, 'test_feat.csv', '3');
saveTest(saveCell{4, 2}, 'test_feat.csv', '4');
saveTest(saveCell{5, 2}, 'test_feat.csv', '5');
end