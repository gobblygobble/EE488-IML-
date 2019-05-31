clear;clc;close all;
%% Your parameter here %%
k = 200; % The number of PCs.

%% Codes %%
[train_matrix,test_matrix] = createDataset();
[project_train_img, k_eig_vec, m] = train_PCA(train_matrix,k);
[recon_error] = train_recon(train_matrix,project_train_img, k_eig_vec,m);
[project_test_img] = test_PCA(test_matrix,k_eig_vec,m);
[id] = identify(project_train_img,project_test_img);