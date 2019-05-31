function [recon_error]=train_recon(train_matrix,eigen_face,k_eig_vec, m)

%% Arguments %%
% train_matrix : training data matrix with dimension of N*d 
%               (N is the number of training images and d is the dimension of one images.)
% project_train_img : training matrix represented by k PCs
% k_eig_vec : k biggest eigen vectors.
% m: mean from training matrix.

%% Your code here %%
% write code to find reconstructed image 'recon_img' and reconstruction
% error.
% recon_img: N*d-dim
recon_img = eigen_face * k_eig_vec';
% reconstruction MSE
mean_vec = zeros(size(train_matrix, 1), size(train_matrix, 2));
for i = 1:size(train_matrix, 1)
    mean_vec(i, :) = m';
end
recon_error = sum(sum((recon_img - mean_vec).^2), 2) / size(train_matrix, 1);
%% save reconstructed images in the folder 'train_reconstruction'
face = zeros(64,64);
mkdir('train_reconstruction')
for i=1:size(recon_img,1)
    fname = sprintf('train_reconstruction/%dres.jpg',i);
    face(:) = recon_img(i,:);
    imwrite(uint8(face), fname);
end
end

