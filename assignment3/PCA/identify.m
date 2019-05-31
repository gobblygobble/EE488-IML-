function [recognized_img] = identify(projectimg,projtestimg)

%% Arguments %%
% project_train_img : training matrix represented by k PCs
% project_test_img : test matrix represented by k PCs

%% Your code here %%
% Hint: Use Euclidean distance to measure similarity.

min_ind = 0;
min = Inf;
for i=1:size(projectimg, 1)
    dist = 0;
    train_mat = projectimg(i, :);
    for j = 1:size(projtestimg, 1)
        diff = projtestimg(j, :) - train_mat;
        dist = dist + sum(diff.^2, 2);
    end
    if dist < min
        min = dist;
        min_ind = i;
    end
end
recognized_img = min_ind;
end
