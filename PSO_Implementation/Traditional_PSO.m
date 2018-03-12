
%% Traditional Path Swarm Optimization
% Pavel Berezovsky
% Meng Project
 
%% PSO 
function Global_Optimal_Solution = Traditional_PSO(Func,Population,number_of_variables,Iterations)
    % Upper and lower bounds of the function chosen
    [Lower_Bound, Upper_Bound, ~] = cost_function(zeros(1,number_of_variables),number_of_variables,Func);
 
    % Setting Upper and Lower Bounds for position and velocity vector
    Bounds_p = [Lower_Bound Upper_Bound];
    Bounds_v = [-(Bounds_p(2)-Bounds_p(1)) Bounds_p(2)-Bounds_p(1)];
    Global.best_cost = inf; % The initial best cost
    Global.best_position = inf; % Initial best position
 
    for i = 1:Population
        % Setting initial position of each particle within the bounds of Func
        Particle(i).position = Bounds_p(1) + 2*Bounds_p(2)*rand(1,number_of_variables);
        % Best position of each particle is its current initial position
        Particle(i).best_position = Particle(i).position; 
        % Obtaining best cost of each particle
        [~,~,Particle(i).best_cost]= cost_function(Particle(i).position,number_of_variables,Func); 
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
            U_Bounds_Check = find(Particle(i).velocity > Bounds_v(2));
            D_Bounds_Check = find(Particle(i).velocity < Bounds_v(1));
            Particle(i).velocity(U_Bounds_Check) = Bounds_v(2);
            Particle(i).velocity(D_Bounds_Check) = Bounds_v(1);
            % Equation for particle position
            Particle(i).position = Particle(i).position+Particle(i).velocity;
            % If particle position exceeds bounds it is set to bounds
            U_Bounds_Check = find(Particle(i).position > Bounds_p(2));
            D_Bounds_Check = find(Particle(i).position < Bounds_p(1));
            Particle(i).position(U_Bounds_Check) = Bounds_p(2);
            Particle(i).position(D_Bounds_Check) = Bounds_p(1);
         
            % Finding temporary cost of the particle
            [~,~,Temp_cost] = cost_function(Particle(i).position,number_of_variables,Func);
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
        % Incrimenting itterations 
        count = count+1;
        
    end
    Global_Optimal_Solution = Global.best_cost;
end
 
