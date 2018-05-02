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

% Calling ISODATA Algorithm
[Generations_needed, number_of_clusters, Cluster, Particle] = Adaptive_PSO(Population, num_variables, Iterations);

Generations_needed
number_of_clusters

% Obtaining best points from each cluster
[X_0, X_0_eval] = Best_points(Cluster,Particle,num_variables);

X_0_eval

[xb, xb_eval] = Black_box_solver(X_0, num_variables);

xb_eval

% Addding matricies above to best point cell arrays
%BP{1,k} = BestPoint;
% Clearing matricies for next interation
%BestPoint = [];


