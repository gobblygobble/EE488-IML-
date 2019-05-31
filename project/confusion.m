function confusionMat = confusion()
%% get kernel and get validity
kernel = getKernel();
x_valid = csvread('valid_feat.csv');
estimate = validation(kernel, x_valid);
truth = csvread('valid_label.csv');
range_val = 10;
%% generate and print confusion matrix
confusionMat = zeros(range_val, range_val);
cMat = zeros(range_val, range_val);

for i = 1 :size(truth, 1)
    cMat(estimate(i, 1), truth(i, 1)) = cMat(estimate(i, 1), truth(i, 1)) + 1;
end
disp(cMat);
class_sum = sum(cMat);
for row = 1 : range_val
    for col = 1 : range_val
        confusionMat(row, col) = round(cMat(row, col) / class_sum(col) * 100);
    end
end
%% plot image
fit = confusionMat./100 * 64;
image(fit);
colormap('gray');
axis equal