%This code uses interior point method of CVX to get the optimal value of
%the LASSO minimization problem corresponding to the dataset generated from 
%generate_dataset_LASSO

clc;
load('SVM_dataset.mat');  %Load the (common) generated dataset

cvx_begin

    variable x(num_dims)
    %minimize mean(max(1 - all_data(:,num_dims+1).*(all_data(:,1:num_dims)*x),0)) + 1/(2*num_data)*(x'*x)
    t = max(1 - all_data(:,num_dims+1).*(all_data(:,1:num_dims)*x),0);
    %minimize 1/num_data*sum(t) + 1/(2*num_data)*quad_over_lin(x,1)
    minimize 1/num_data*sum(t) + 1/(2*local_dataset_size)*quad_over_lin(x,1)
    
cvx_end

%Save the solution obtained from CVX
save('SVM_cvx_output.mat');
