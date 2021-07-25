%This code solves the problem of decentralized SVM using DLMD DiffEx
%(Decentralized Lazy Mirror Descent with Differential Exchanges) over a
%noisy network under finite data rate constraints.

%For a fixed time horizon, it empirically obtains the tradeoff between
%suboptimality gap and success probability

clc;
load('SVM_dataset.mat');  %Load the (common) generated dataset

max_iters = 75;           %Time horizon after which the suboptimality gap is obtained
num_trials = 100;         %No. of times the algorithm is run for each value of dynamic range

%Network connectivity structure
P = (1/num_nodes)*ones(num_nodes, num_nodes);               %Fully connected network

%Ring network (with specified number of neighbors)
% num_ngbrs = 1;
% P = zeros(num_nodes, num_nodes);
% for i = 1:1:num_nodes
%     P(i,i) = 1/(2*num_ngbrs+1);
%     for j = 1:1:num_ngbrs
%         
%         %Right neighbors
%         if(i+j <= num_nodes)
%             P(i,i+j) = 1/(2*num_ngbrs+1);
%         else
%             P(i, rem(i+j,num_nodes)) = 1/(2*num_ngbrs+1);
%         end
%         
%         %Left neighbors
%         if(i-j >= 1)
%             P(i,i-j) = 1/(2*num_ngbrs+1);
%         else
%             P(i, i-j+num_nodes) = 1/(2*num_ngbrs+1);
%         end
%         
%     end
% end

%Dynamic range variation
min_quantizer_range = 0.8;
max_quantizer_range = 1.8;
quantizer_range_increment = 0.1;

%Success probability for each value of quantizer range
dynamic_range_variation_length = ceil((max_quantizer_range-min_quantizer_range)/quantizer_range_increment) + 1;
success_prob = zeros(dynamic_range_variation_length, 1);    
U_counter = 0;          %To index success_prob array

%Power control parameters
kappa = 0;              %Power scaling
gamma = 0.5;              %Diminishing confidence rate

%For storing the objective function value of successful runs
obj_fnval_avg = zeros(num_nodes, max_iters, dynamic_range_variation_length);

%Outer loop for varying quantizer range
for U = min_quantizer_range:quantizer_range_increment:max_quantizer_range
    
    U_counter = U_counter + 1;
    
    num_success = 0;        %No. of times the algorithm succeeded
    
    %Quantizer parameters
    R = 3;                      %Data rate per dimension
    M = 2^R;                    %No. of quantization levels
    DELTA = 2*U/(M-1);          %Quantization resolution
    
    %Repeating the algorithm execution num_trials number of times for a
    %fixed value of Quantizer range U to see how many times it succeeds
    for trial_idx = 1:1:num_trials
        
        %Flag to certify if algorithm succeeds
        Success = true;
        
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
        
        %Run the algorithm
        for k = 1:1:max_iters
            
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
            noise_power = 0.05;
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
            eta = 0.5/(sqrt(noise_power^2*num_dims/(2*gamma) + 0.25*DELTA^2*num_dims)*k^((1 + gamma)/2));
            %eta = 0.5/(noise_power*k^((1 + gamma)/2));          %Step size schedule
            %eta = 0.5/sqrt(k);
            for i = 1:1:num_nodes
                x(:,i) = -eta*z(:,i);
            end

            %Update running average
            x_av = (1/k)*((k-1)*x_av + x);
            
            %Store normed hinge loss for plotting later
            for i = 1:1:num_nodes
               obj_fnval(i,k) = mean(max(1 - all_data(:,num_dims+1).*(all_data(:,1:num_dims)*x_av(:,i)),0)) + 1/(2*local_dataset_size)*norm(x_av(:,i),2)^2;
            end

            %Store classification error
            for i = 1:1:num_nodes
                pred = sign(all_data(:,1:num_dims)*x_av(:,i));
                classification_error(i,k) = mean(pred.*all_data(:,num_dims+1) < 0);
            end
            
            if(max_omega > U || min_omega < -U)
                Success = false;
                break;
            end
           
        end
        
        %Check if the algorithm succeeded and increment count
        if (Success == true)
            obj_fnval_avg(:,:,U_counter) = obj_fnval_avg(:,:,U_counter) + obj_fnval;
            num_success = num_success + 1;
        end
            
    end
    
    %Computing the empiricial probability of success
    success_prob(U_counter) = num_success/num_trials;
    
    %Compute the realization averaged objective function value
    obj_fnval_avg(:,:,U_counter) = obj_fnval_avg(:,:,U_counter)/num_success;
    
    %To keep track of progress
    fprintf('\nU = %f\n', U);
    fprintf('Success probability for U = %f is %f\n', U, num_success/num_trials);
    fprintf('max_omega = %f\n', max_omega);
    fprintf('min_omega = %f\n', min_omega);
    
end

%Save and plot the final objective function value
obj_fnval_dynamic_range_variation = squeeze(obj_fnval_avg(1,max_iters,:));
plot(success_prob, obj_fnval_dynamic_range_variation-cvx_optval);

%%

figure;
dynamic_range_var = min_quantizer_range:quantizer_range_increment:max_quantizer_range;
plot(dynamic_range_var, success_prob_ring, '-x', 'LineWidth', 1.5);
title('SVM','fontweight','bold','fontsize',30);
hold on;
grid on;
plot(dynamic_range_var, success_prob_FC, '-o', 'LineWidth', 1.5);
xlabel('\bf Quantizer dynamic range (U)', 'FontSize', 26);
ylabel('\bf Success probability P_S(K)', 'FontSize', 30);
lgd = legend('\bf Ring network', '\bf Fully-connected network');
lgd.FontSize = 20;

%%
figure;
plot(success_prob_ring, obj_fnval_dynamic_range_variation_ring-cvx_optval, '-x', 'LineWidth', 2);
title('SVM','fontweight','bold','fontsize',30);
hold on;
grid on;
plot(success_prob_FC, obj_fnval_dynamic_range_variation_FC-cvx_optval, '-o', 'LineWidth', 2);
xlabel('\bf Success probability P_S(K)', 'FontSize', 26);
ylabel('\bf f(x_1(K)) - f_*', 'FontSize', 30);
lgd = legend('\bf Ring network', '\bf Fully-connected network');
lgd.FontSize = 20;

