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
Iterations = 1000;

% Calling Traditional_PSO Algorithm
[GlobalBest, GlobalPos, Total] = Traditional_PSO(Population, num_variables, Iterations)

%[xb, xb_eval] = Black_box_solver(X_0, num_variables);

%xb_eval

% Addding matricies above to best point cell arrays
%BP{1,k} = BestPoint;
% Clearing matricies for next interation
%BestPoint = [];


