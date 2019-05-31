function [project_train_img, k_eig_vec, m] = train_PCA(train_matrix,k)

%% Arguments %%
% train_matrix : training image matrix with dimension of N*d 
%               (N is the number of training images and d is the dimension of one images.)
% k: number of PCs

%% Your code here %%
% write code to find eigen vectors 'eigen_face'.
% covariance matrix of train matrix X: d*d dimension matrix transpose(X)*X
cov_mat = train_matrix' * train_matrix;
% compute diagonal matix with eigenvalues on its diagonal D and the
% matrix of corresponding eigenvectors V
[V, D] = eig(cov_mat);
% sort the eigenvalues and change the corresponding eigenvectors as well
[~, I] = sort(diag(D), 'descend');
V = V(:, I);
% save the k biggest eigenvectors in k_eig_vec
k_eig_vec = V(:, 1:k);
% training matrix represented by the k PCs: dimension N*k
project_train_img = train_matrix * k_eig_vec;
% compute and save mean of training matrix
m = sum(train_matrix)' / size(train_matrix, 1);
%% display 3 biggest eigen vectors
for i = 1:3
    subplot(1,3,i)
    imagesc(reshape(k_eig_vec(:,i), 64, 64));
end

end
