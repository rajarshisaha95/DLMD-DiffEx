%This code uses interior point method of CVX to get the optimal value of
%the LASSO minimization problem corresponding to the dataset generated from 
%generate_dataset_LASSO

clc;
load('LASSO_dataset.mat');  %Load the (common) generated dataset

lambda = 0.1;             %LASSO regulatization parameter

cvx_begin

    variable x(num_dims)
    minimize (1/(2*num_data)*(y - A*x)'*(y - A*x) + lambda*norm(x,1));

cvx_end

%Save the solution obtained from CVX
save('LASSO_cvx_output.mat');
