function theta = train(x_train,y_train,T)
%% Arguments %%
% x_train: training data
% y_train: training label
% T: the number of iteration
% theta: weight parameter
%% Your code here %%
% set lambda to 1
lambda = 1;
% dimension d
d = size(x_train, 2);
% initialize w, changes over iteration
w = zeros(1,d);
% theta_sum, the sum of all theta_i in loop
theta_sum = zeros(1,d);
for t = 1:T
    theta_t = (1/(lambda * t)) * w;
    % random number i
    i = randi(d);
    x_i = x_train(i,:);
    y_i = y_train(i,1);
    if y_i * dot(theta_t, x_i) < 1
        w = w + y_i * x_i;
    end
    % else w = w;
    theta_sum = theta_sum + theta_t;
end
% theta = average of theta_i
theta = theta_sum / T;
end