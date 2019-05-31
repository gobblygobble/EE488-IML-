function [acc] = test(x_test,y_test,theta)
%% Argurments %%
% x_test: test data
% y_test: test label
% theta: weight parameter
% acc: accuracy
%% Your code here %%
% number of correct estimates cor
% estimate label y_hat
cor = 0;
for it = 1:size(x_test,1)
    x_it = x_test(it,:);
    y_hat = sign(dot(x_it, theta));
    if y_hat == y_test(it,1)
        cor = cor + 1;
    end
end
acc = cor / size(x_test, 1);
end