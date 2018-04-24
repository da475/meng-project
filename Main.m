%% ISODATA Optimization Algorithm Main File
% Pavel Berezovsky
% Meng Project
 
%% Main
close all; clc;
 
% INITIALIZATION
% Initilize different values of func based on cost function to be tested
% Func = 1 -> Rastrigin Function
% Func = 2 -> Shfited Sphere Function
% Func = 3 -> Griewank Function
% Func = 4 -> Shifted Rosenbrock
% Func = 5 -> Shifted Rotated Ackley

BestPoint = []; % Saving Best points for 1 function call
% Saving the matricies above in a storage
BP = {};   

Population = 20;
num_variables = 3;
Iterations = 20000;
Generations_needed = zeros(1,15);
number_of_clusters = zeros(1,15);
for k = 1:2
    for i = 1:5
        Func = i;
        disp(['Loop ' num2str(k) '  Function ' num2str(i)])
        % Calling ISODATA Algorithm
        [Generations_needed(i,k),number_of_clusters(i,k),Cluster,Particle] = ISODATA_PSO(Func,Population,num_variables,Iterations);
        % Obtaining best points from each cluster
        X_0 = Best_points(Cluster,Particle,num_variables,Func); 
        % Saving best points into Best point matrix
        BestPoint = [BestPoint;X_0;zeros(1,num_variables)];
    end
    % Addding matricies above to best point cell arrays
    BP{1,k} = BestPoint;
   % Clearing matricies for next interation
    BestPoint = [];
end


