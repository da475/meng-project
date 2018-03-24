%% Traditional Optimization Algorithm Main File
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

Population = 50;
num_variables = 2;
Iterations = 2000;
Data = zeros(5,2);
for i = 1:5
    Func = i;
    disp(['Function ' num2str(i)])
    for k = 1:2
         Data(i,k) = Traditional_PSO(Func,Population,num_variables,Iterations);
    end
    Mean(i,1) = mean(Data(i,:));
    Std(i,1) = std(Data(i,:));
end

