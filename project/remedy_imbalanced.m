function [balanced_x, balanced_y] = remedy_imbalanced(imbalanced_x, imbalanced_y)
%% remedies the imbalanced data problem by taking the same number of data for each of the indices
% parameters
index_num = 10;
% sort in ascending order
[sort_y, I] = sort(imbalanced_y, 'ascend');
sort_x = imbalanced_x(I, :);
% count the number of each index
count = zeros(index_num, 1);
for ind = 1 : size(sort_y, 1)
    count(sort_y(ind, 1), 1) = count(sort_y(ind, 1), 1) + 1;
end
min_count = min(count);
total_count = index_num * min_count;
balanced_x = zeros(total_count, size(imbalanced_x, 2));
balanced_y = zeros(total_count, 1);
% choose random data
for i = 1 : index_num
    i_x = sort_x((sum(count(1:i - 1)) + 1):sum(count(1:i)), :);
    perm_i_x = randperm(count(i, 1));
    for count_ind = 1 : min_count
        balanced_x((i - 1) * min_count + count_ind, :) = i_x(perm_i_x(count_ind), :);
        balanced_y((i - 1) * min_count + count_ind, 1) = i;
    end
end
end
