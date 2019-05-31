function theta = train(x_train,y_train,eta,T)
%% Arguments %%
% x_train: training data
% y_train: training label
% eta: learning rate
% T: the number of iteration
% theta: weight parameter
%% Your code here %%
% number of training data n
% dimension of each training data d
% estimate label y_hat
% num-th data and label x_num and y_num
n = size(x_train, 1);
d = size(x_train, 2);
theta = zeros(1,d);
% repeat T times
for it = 1:T
    for num = 1:n
        x_num = x_train(num,:);
        y_num = y_train(num,:);
        y_hat = sign(dot(x_num, theta));
        if y_hat == 0
            y_hat = 1;
        end
        if y_hat ~= y_num
            theta = theta + eta * y_num * x_num;
        end
    end
end
end