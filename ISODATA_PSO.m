
%% ISODATA Path Swarm Optimization
% Pavel Berezovsky
% Meng Project
 
%%  Concensus Based PSO 
function [count,nc,Cluster,Particle] = ISODATA_PSO(Population,number_of_variables,Iterations)
    
    % Pre-defined variables
    
    Cluster_num = 6; % number of clusters at the start of the algorithm
    Desired_cluster = 4; % desired number of clusters
    Initial_clusters = randperm(Population,Cluster_num); % random index in population to save particle positions as initial cluster centers
    Cluster_min_memb = 4; % minimum members in a cluster
    Cluster_max_std = .01; % cluster maximum standard deviation for splitting
    Cluster_min_dis = 1; % cluster minimum distance for merging
    Cluster = cell(Cluster_num,2);
    
    Test1 = {};
    Test2 = {};
    Test3 = {};
          
    % Upper and lower bounds of the function chosen
    Lower_Bound = [0.01, 0.1, 0.5]
    Upper_Bound = [0.5, 100, 5]

    % Setting Upper and Lower Bounds for position and velocity vector
    Bounds_p = [Lower_Bound; Upper_Bound];
    Bounds_v = [-(Bounds_p(2, :)-Bounds_p(1, :)); Bounds_p(2, :)-Bounds_p(1, :)];

    Global.best_cost = inf; % The initial best cost
    Global.best_position = inf; % Initial best position
 
    for i = 1:Population
        % Setting initial position of each particle within the bounds of Func
        Particle(i).position = Bounds_p(1, :) + ((Bounds_p(2, :) - Bounds_p(1, :)).*rand(1,number_of_variables));
        % Best position of each particle is its current initial position
        Particle(i).best_position = Particle(i).position; 
        
		% Obtaining best cost of each particle
        cmd = num2str(Particle(i).position);
		cmd = ['python svm_model.py ' cmd];

		[~, result] = system(cmd);
		Particle(i).best_cost = str2num(result);
        
		% If best cost of each particle is less than the global cost set
        % global cost
        if Particle(i).best_cost < Global.best_cost
            Global.best_cost = Particle(i).best_cost;
            Global.best_position = Particle(i).best_position;
        end
        % Initial particle velocity is 0
        Particle(i).velocity = zeros(1,number_of_variables);
    end
 
    % PSO LOOP
    % iteration count
    count = 0;
    % Inertial coefficient (memory)
    w = .8; 
    % Coefficients 
    c1 = 2; c2 = 2;
 
    while (count < Iterations)
        for i = 1:Population
            % Random elements added to generate velocity
            r1 = rand;
            r2 = rand;
            % Equation of particle velocity
            Particle(i).velocity = w*Particle(i).velocity...
             +c1*r1*(Particle(i).best_position-Particle(i).position)...
             +c2*r2*(Global.best_position-Particle(i).position);
            % If particle velocity exceeds bounds it is set to bounds
            U_Bounds_Check = find(Particle(i).velocity > Bounds_v(2, :));
            D_Bounds_Check = find(Particle(i).velocity < Bounds_v(1, :));
            Particle(i).velocity(U_Bounds_Check) = Bounds_v(2, U_Bounds_Check);
            Particle(i).velocity(D_Bounds_Check) = Bounds_v(1, D_Bounds_Check);
            % Equation for particle position
            Particle(i).position = Particle(i).position+Particle(i).velocity;
            % If particle position exceeds bounds it is set to bounds
            U_Bounds_Check = find(Particle(i).position > Bounds_p(2, :));
            D_Bounds_Check = find(Particle(i).position < Bounds_p(1, :));
            Particle(i).position(U_Bounds_Check) = Bounds_p(2, U_Bounds_Check);
            Particle(i).position(D_Bounds_Check) = Bounds_p(1, D_Bounds_Check);
         
            % Finding temporary cost of the particle
			cmd = num2str(Particle(i).position);
			cmd = ['python svm_model.py ' cmd];

			[~, result] = system(cmd);
			Temp_cost = str2num(result);

            % If temporary cost is less than particle best cost
            if Temp_cost < Particle(i).best_cost
                Particle(i).best_cost = Temp_cost;
                Particle(i).best_position = Particle(i).position;
            end
            % If global best cost is greater than particle best cost 
            if Particle(i).best_cost < Global.best_cost
                Global.best_cost = Particle(i).best_cost;
                Global.best_position = Particle(i).best_position;
            end 
        end
        % Tracking global cost
        Total(count+1) = Global.best_cost;
        % ISODATA STARTS HERE
        if (count == 300) % First call to ISODATA after 300 iterations of PSO    
            for i = 1:Cluster_num
                Cluster{i,1} = Particle(Initial_clusters(i)).position; % Initialize cluster centers as random particle positions
            end
          % Calling ISODATA Algorithm
          Cluster =ISODATA_new(Particle,Cluster,Cluster_num,Cluster_min_memb,Cluster_max_std,Cluster_min_dis,number_of_variables,Desired_cluster);     
          Test3 = Test2;
          Test2 = Test1;
          Test1 = Cluster;
        end 
        if (count > 300) && (mod(count,20) == 0) % Calling ISODATA every 20 iterations after the 300th iteration
             Cluster =ISODATA_new(Particle,Cluster,Cluster_num,Cluster_min_memb,Cluster_max_std,Cluster_min_dis,number_of_variables,Desired_cluster); 
             Test3 = Test2;
             Test2 = Test1;
             Test1 = Cluster;
        end
        Result = Algorithm_termination(Test1,Test2,Test3); % Check for ISODATA convergence, if Result = 1 then terminate algorithm, else continue
        if Result == 1
             nc = length(Test1(:,1));
             break;
        end
        % Empty second column of cluster that contains members belonging to that cluster
        for i = 1:length(Cluster(:,1))
            Cluster{i,2} = [];
        end
        Cluster_num = length(Cluster(:,1)); % Modify cluster number for next run of ISODATA      
             % Incrimenting itterations 
        count = count+1
        
    end
    
end
 
