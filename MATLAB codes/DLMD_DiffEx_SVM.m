%This code solves the problem of decentralized SVM using DLMD-DiffEx 
%(Decentralized Lazy Mirror Descent with Differential Exchanges) over a
%noisy network under finite-data rate constraints

clc;
load('SVM_dataset.mat');  %Load the (common) generated dataset
load('SVM_cvx_output.mat');

num_realizations = 5;       %No. of ensemble realizations for which the results are averaged
max_iters = 10000;            %Maximum number of iterations in each realization

%Network connectivity structure
%P = (1/num_nodes)*ones(num_nodes, num_nodes);               %Fully connected network

%Ring network (with specified number of neighbors)
num_ngbrs = 1;
P = zeros(num_nodes, num_nodes);
for i = 1:1:num_nodes
    P(i,i) = 1/(2*num_ngbrs+1);
    for j = 1:1:num_ngbrs
        
        %Right neighbors
        if(i+j <= num_nodes)
            P(i,i+j) = 1/(2*num_ngbrs+1);
        else
            P(i, rem(i+j,num_nodes)) = 1/(2*num_ngbrs+1);
        end
        
        %Left neighbors
        if(i-j >= 1)
            P(i,i-j) = 1/(2*num_ngbrs+1);
        else
            P(i, i-j+num_nodes) = 1/(2*num_ngbrs+1);
        end
        
    end
end

%Quantizer parameters
R = 6;                      %Data rate per dimension
U = 100;                     %Specifies the range of the quantizer [-U,+U]
M = 2^R;                    %No. of quantization levels
DELTA = 2*U/(M-1);          %Quantization resolution

%Power control parameters
kappa = 0;              %Power scaling
gamma = 0.5;               %Diminishing confidence rate

%To store average over different ensemble realizations
obj_fnval_avg = zeros(num_nodes, max_iters);                %Each row corresponds to a different node
classification_error_avg = zeros(num_nodes, max_iters);     %For plotting the classification error instead of the objective function value

for realiz_ind = 1:1:num_realizations
    
    %Initial states
    z = zeros(num_dims, num_nodes);                         %Dual iterates
    z_new = zeros(num_dims, num_nodes);
    x = x_init;                                             %Primal iterates
    x_av = x_init;                                          %Iteration averaged primal iterate
    
    %Global objective (and classification error) evaluated at the iterates of each node
    obj_fnval = zeros(num_nodes, max_iters);
    classification_error = zeros(num_nodes, max_iters);
    
    %Proxy for past state of a node being maintained by itself (for another node - 2nd index)
    %Not required to keep separate proxies for different nodes because all
    %links have same data rate constraint (but still doing it)
    %yp(:,j,i) is the proxy for node i's state (stored by itself) for node j
    yp = zeros(num_dims, num_nodes, num_nodes);
    yp_tilde = zeros(num_dims, num_nodes, num_nodes);       %yp_tilde(:,j,i) means the estimate of node j's state maintained by node i

    %omega(:,j,i) stores the differential encoded by node i for node j
    omega = zeros(num_dims, num_nodes, num_nodes);

    %delta(:,j,i) stores the quantized differential encoded by node i for node j
    delta = zeros(num_dims, num_nodes, num_nodes);
    
    %Gradients. Each column corresponds to the gradient of a particular node
    grad = zeros(num_dims, num_nodes);
    
    %To check saturation
    max_omega = -Inf;
    min_omega = +Inf;
    
    for k = 1:1:max_iters
        
        k
        
        %Store normed hinge loss for plotting later
        for i = 1:1:num_nodes
           obj_fnval(i,k) = mean(max(1 - all_data(:,num_dims+1).*(all_data(:,1:num_dims)*x_av(:,i)),0)) + 1/(2*local_dataset_size)*norm(x_av(:,i),2)^2;
        end
        
        %Store classification error
        for i = 1:1:num_nodes
            pred = sign(all_data(:,1:num_dims)*x_av(:,i));
            classification_error(i,k) = mean(pred.*all_data(:,num_dims+1) < 0);
        end
        
        %Compute and quantize differential at each node for all other nodes
        for i = 1:1:num_nodes
            for j = 1:1:num_nodes
                
                %Compute differentials
                omega(:,j,i) = z(:,i) - yp(:,j,i);
                
                %Quantize differentials
                for ii = 1:1:num_dims

                    K_t = floor((omega(ii,j,i)+U)/DELTA);       %Index of lower quantization value
                    lower_point = -U + K_t*DELTA;               %Lower quantization value
                    upper_point = -U + (K_t+1)*DELTA;           %Upper quantization value

                    p = (omega(ii,j,i)-lower_point)/DELTA;      %Probability of quantizing to the upper value

                    if(rand > p)
                        delta(ii,j,i) = lower_point;            %Quantize to lower value with probability (1-p)
                    else
                        delta(ii,j,i) = upper_point;            %Quantize to upper value with probability p
                    end                
                end
                
            end
        end
        
        %Track the maximum of differentials
        maximum = max(max(max(omega)));
        if (max_omega < maximum)
            max_omega = maximum;
        end
    
        %Track the minimum of differentials
        minimum = min(min(min(omega)));
        if (min_omega > minimum)
            min_omega = minimum;
        end
        
        %Update proxies (by adding the quantized differential)
        for i = 1:1:num_nodes
            for j = 1:1:num_nodes
                yp(:,j,i) = yp(:,j,i) + delta(:,j,i);
            end
        end
        
        %Transmit over noisy channel and decode
        alpha = k^(kappa/2);            %Power scaling
        
        %Transmit signal
        tx_signal = alpha*delta;
        
        %Received signal
        noise_power = 0.1;
        rx_signal = tx_signal + noise_power*randn(size(delta));
        delta_tilde = rx_signal./alpha;
        
        %Nodes decode running estimates of neighbors states
        for i = 1:1:num_nodes
            for j = 1:1:num_nodes
                yp_tilde(:,j,i) = yp_tilde(:,j,i) + delta_tilde(:,i,j);
            end
        end
        
        %Evaluate subgradient for each node
        for i = 1:1:num_nodes           
            %Randomly sampling datapoint 
            frac_data = 0.7;                %Fraction of local dataset used for computing (sub)gradient
            num_local_indices = floor(frac_data*local_dataset_size);
            local_idx = randperm(local_dataset_size, num_local_indices);
            sampled_local_data = dataset(local_idx, :, i);
            
            %Computing stochastic (sub)gradient
            is_active = sampled_local_data(:,num_dims+1).*(sampled_local_data(:,1:num_dims)*x_av(:,i)) < 1;
            
            temp = zeros(num_local_indices, num_dims);
            for j = 1:1:num_local_indices
                temp(j,:) = is_active(j)*(sampled_local_data(j,num_dims+1)*sampled_local_data(j,1:num_dims));
            end
            grad(:,i) = -mean(temp, 1)' + x_av(:,i)/local_dataset_size;    
        end
        
        beta = 1/k^gamma;           %Diminishing confidence
        
        %Nodes update states
        for i = 1:1:num_nodes
            
            W = (1-beta)*eye(num_nodes) + beta*P;
            t = W(i,i)*z(:,i);
            for j = 1:1:num_nodes
                if (j == i);
                else
                    t = t + W(i,j)*yp_tilde(:,j,i);
                end
            end
            
            %Taking a subgradient step
            z_new(:,i) = t + grad(:,i);
            
        end
        
        z = z_new;
        
        %Projection (onto primal space) for unconstrained optimization with squared proximal function
        eta = 1/(sqrt(noise_power^2*num_dims/(2*gamma) + 0.25*DELTA^2*num_dims)*k^((1 + gamma)/2));
        %eta = 1/(sqrt(0.25*DELTA^2*num_dims)*k^((1 + gamma)/2));
        %eta = 0.5/(noise_power*k^((1 + gamma)/2));    %Dont use
        %eta = 0.5/sqrt(k);                            %Dont use
        for i = 1:1:num_nodes
            x(:,i) = -eta*z(:,i);
        end
        
        %Update running average
        x_av = (1/k)*((k-1)*x_av + x);
              
    end
    
    obj_fnval_avg = obj_fnval_avg + obj_fnval;
    classification_error_avg = classification_error_avg + classification_error;
    
    fprintf('\n(Decentralized SVM) Realization: %d\n', realiz_ind);
  
end

obj_fnval_avg = obj_fnval_avg/num_realizations;
classification_error_avg = classification_error_avg/num_realizations;

obj_fnval_avg(1,max_iters)

%%
figure;
[aline(1), aFill(1)] = stdshade(obj_fnval_avg_noiseless-cvx_optval, 0.3 , 'k');
title('SVM:Fully-connected Network','fontweight','bold','fontsize',30);
xlabel('Iterations(K)','fontweight','bold','fontsize',24);
ylabel('Suboptimality gap f(x_{av}(K)) - f_*','fontweight','bold','fontsize',24);
hold on;
grid on;
[aline(4), aFill(4)] = stdshade(obj_fnval_avg_gamma_pt5-cvx_optval,0.3,'g');
[aline(2), aFill(2)] = stdshade(obj_fnval_avg_noisy-cvx_optval,0.3,'r');
[aline(3), aFill(3)] = stdshade(obj_fnval_avg_gamma_pt1-cvx_optval,0.3,'b');
lgd = legend(aline,'\bf Noiseless','\bf Noisy without confidence', '\bf \gamma = 0.1, \kappa = 0.8', '\bf \gamma = 0.5, \kappa = 0');
lgd.FontSize = 20;


