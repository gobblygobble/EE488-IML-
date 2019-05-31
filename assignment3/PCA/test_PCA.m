function [project_test_img] = test_PCA(test_matrix,k_eig_vec,m)

%% Arguments %%
% test_matrix : test image matrix with dimension of N*d 
%               (N is the number of training images and d is the dimension of one images.)
% k_eig_vec : k biggest eigen vectors from training matrix.
% m: mean from training matrix.

%% Your code here %%

project_test_img = test_matrix * k_eig_vec;

end
