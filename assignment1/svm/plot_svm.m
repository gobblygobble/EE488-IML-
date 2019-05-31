function plot_svm(x_train,y_train,x_test,y_test,theta)
%% Arguments %%
% x_train: training data
% y_train: training label
% x_test: test data
% y_test: test label
% theta: weight parameter
%% Your code here %%
% same code as /svm/test.m
x_1 = linspace(0, 10);
theta_1 = theta(1,1);
theta_2 = theta(1,2);
slope = -theta_1 / theta_2;
x_2 = slope * x_1;
% plot decision boundary
plot(x_1, x_2);
hold on;
% plot training data
for num = 1:size(x_train, 1)
    x1 = x_train(num,1);
    x2 = x_train(num,2);
    y = y_train(num,1);
    if y < 0
        plot(x1, x2, 'bo');
    else
        plot(x1, x2, 'ro');
    end
end
% plot test data
for num = 1:size(x_test, 1)
    x1 = x_test(num,1);
    x2 = x_test(num,2);
    y = y_test(num,1);
    if y < 0
        plot(x1, x2, 'bo');
    else
        plot(x1, x2, 'ro');
    end
end
end