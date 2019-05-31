function K = svmkernel(x_i, x_j,kernel_type)
%% Arguments %%
% x_i: ith random data from [m]
% x_j: j ~= i data from [m]
% kernel_type: kernel types : 'linear' or 'rbf'
%% Your code here %%
    switch kernel_type
      case 'linear'
          % K = ;
      case 'rbf'
          % K = ;
      otherwise
        error('Unknown kernel function');
      
      K = double(K);
    end
end
% Convert to full matrix if inputs are sparse