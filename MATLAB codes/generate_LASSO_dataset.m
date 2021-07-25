%This script generates dataset for decentralized regression and stores it
%for a fair comparison across different problem settings.

clear all;
close all;
clc;

num_nodes = 10;             %No. of nodes in the network
num_dims = 30;              %Dimension of the problem
num_data = 2000;            %Total no. of data points

local_dataset_size = num_data/num_nodes;                %Size of local dataset on each node

%Generate complete LASSO regression data
A = randn(num_data, num_dims);             
y = randn(num_data, 1);                 

x_opt = (A'*A)\(A'*y);       %Optimal solution in closed form

%Splitting the dataset across nodes
A_dist = zeros(local_dataset_size, num_dims, num_nodes);     %The last index indexes the nodes
y_dist = zeros(local_dataset_size, num_nodes);               %Output

for i = 1:1:num_nodes
    
    A_dist(:,:,i) = A((i-1)*local_dataset_size+1:i*local_dataset_size,:);
    y_dist(:,i) = y((i-1)*local_dataset_size+1:i*local_dataset_size, 1);
    
end

x_init = randn(num_dims, num_nodes);

%Save the dataset
save('LASSO_dataset.mat');
