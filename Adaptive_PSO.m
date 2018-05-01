
%% Adaptive Particle Swarm Optimization
% Pavel Berezovsky
% Meng Project
%% PSO 
function [count,nc,Cluster,Particle_s1] = Adaptive_PSO(Func,Population,number_of_variables,Iterations)
    % Upper and lower bounds of the function chosen
    [Lower_Bound, Upper_Bound, ~] = cost_function(zeros(1,number_of_variables),number_of_variables,Func);
 
    % Setting Upper and Lower Bounds for position and velocity vector
    
    Bounds_p = [Lower_Bound+zeros(1,number_of_variables); Upper_Bound+zeros(1,number_of_variables)];
    Bounds_v = [-(Bounds_p(2,:)-Bounds_p(1,:)); Bounds_p(2,:)-Bounds_p(1,:)];
    Global_s1.best_cost = inf; % The initial best cost
    Global_s1.best_position = inf; % Initial best position
    
    Global_s2.best_cost = inf; % The initial best cost
    Global_s2.best_position = inf; % Initial best position
    Dp_s1 = zeros(1,Iterations); Dp_s2 = zeros(1,Iterations);
    Dp_s1(1) = .5;Dp_s2(1) = .5;
    Fl_s1 = zeros(1,Iterations); Fl_s2 = zeros(1,Iterations);
    q1 = Population/2;q2 = Population/2;
    Extinction_s1 = 0;Extinction_s2 = 0;
    S1_best_it = inf+zeros(1,Iterations);
    S2_best_it = inf+zeros(1,Iterations);
    S1_worst.member = 0;S2_worst.member = 0;S1_worst.value = -inf;S2_worst.value = -inf;
    S1_best.member = 0;S2_best.member = 0;S1_best.value = inf;S2_best.value = inf;
    M = 50;
    for i = 1:Population/2
        % Setting initial position of each particle within the bounds of Func
        Particle_s1(i).position = Bounds_p(1,:) + (Bounds_p(2,:)-Bounds_p(1,:)).*rand(1,number_of_variables);
        % Best position of each particle is its current initial position
        Particle_s1(i).best_position = Particle_s1(i).position; 
        % Obtaining best cost of each particle
        [~,~,Particle_s1(i).best_cost]= cost_function(Particle_s1(i).position,number_of_variables,Func); 
        if(S1_best_it(1) > Particle_s1(i).best_cost)
            S1_best_it(1) = Particle_s1(i).best_cost;
        end
        % If best cost of each particle is less than the global cost set
        % global cost
        if Particle_s1(i).best_cost < Global_s1.best_cost
            Global_s1.best_cost = Particle_s1(i).best_cost;
            Global_s1.best_position = Particle_s1(i).best_position;
        end
        % Initial particle velocity is 0
        Particle_s1(i).velocity = zeros(1,number_of_variables);
    end
    
    for i = 1:Population/2
        % Setting initial position of each particle within the bounds of Func
        Particle_s2(i).position = Bounds_p(1,:) + (Bounds_p(2,:)-Bounds_p(1,:)).*rand(1,number_of_variables);
        % Best position of each particle is its current initial position
        Particle_s2(i).best_position = Particle_s2(i).position; 
        % Obtaining best cost of each particle
        [~,~,Particle_s2(i).best_cost]= cost_function(Particle_s2(i).position,number_of_variables,Func); 
        
        if(S2_best_it(1) > Particle_s2(i).best_cost)
            S2_best_it(1) = Particle_s2(i).best_cost;
        end
        % If best cost of each particle is less than the global cost set
        % global cost
        if Particle_s2(i).best_cost < Global_s2.best_cost
            Global_s2.best_cost = Particle_s2(i).best_cost;
            Global_s2.best_position = Particle_s2(i).best_position;
        end
        % Initial particle velocity is 0
        Particle_s2(i).prevposition = zeros(1,number_of_variables);
    end
    
    % Coefficients 
    c1 = 1.49445;
    w0 = .9;
    w1 = .4;
    c = 4;
    hk = 1;
    count = 2;
    while (count < Iterations)
        if(Extinction_s1 ~= 1)
            % Inertial coefficient (memory)
            w(count) = w0*(w0-w1)*count/Iterations;
            for i = 1:q1
                % Random elements added to generate velocity
                r1 = rand;
                temp_pbest = zeros(1,number_of_variables);
                Pc(i) = 0.05+-.45*(exp(10*(i-1)/q1-1)-1)/(exp(10)-1);
                for j = 1:number_of_variables
                    r4 = rand;
                    if(r4 >= Pc(i))
                        temp_pbest(j) = Particle_s1(i).best_position(j);
                    else
                        test_particle1_num = randi(q1);
                        test_particle2_num = randi(q1);
                        if(rand < .5)
                            test_particle1 = Particle_s1(test_particle1_num).best_position;
                            test_particle2 = test_particle1;
                            test_particle2(j) = Particle_s1(test_particle2_num).best_position(j);
                        else
                            test_particle2 = Particle_s1(test_particle2_num).best_position;
                            test_particle1 = test_particle2;
                            test_particle1(j) = Particle_s1(test_particle1_num).best_position(j);
                        end
                        [~,~,particle1_cost] = cost_function(test_particle1,number_of_variables,Func);
                        [~,~,particle2_cost] = cost_function(test_particle2,number_of_variables,Func);
                        if(particle1_cost < particle2_cost)
                            temp_pbest(j) = test_particle1(j);
                        else
                            temp_pbest(j) = test_particle2(j);
                        end
                    end
                end
                % Equation of particle velocity
                Particle_s1(i).velocity = w(count)*Particle_s1(i).velocity...
                +c1*r1*(temp_pbest-Particle_s1(i).position);
                % If particle velocity exceeds bounds it is set to bounds
                U_Bounds_Check = find(Particle_s1(i).velocity > Bounds_v(2,:));
                D_Bounds_Check = find(Particle_s1(i).velocity < Bounds_v(1,:));
                Particle_s1(i).velocity(U_Bounds_Check) = Bounds_v(2,U_Bounds_Check);
                Particle_s1(i).velocity(D_Bounds_Check) = Bounds_v(1,D_Bounds_Check);
                % Equation for particle position
                Particle_s1(i).position = Particle_s1(i).position+Particle_s1(i).velocity;
                % If particle position exceeds bounds it is set to bounds
                U_Bounds_Check = find(Particle_s1(i).position > Bounds_p(2,:));
                D_Bounds_Check = find(Particle_s1(i).position < Bounds_p(1,:));
                Particle_s1(i).position(U_Bounds_Check) = Bounds_p(2,U_Bounds_Check);
                Particle_s1(i).position(D_Bounds_Check) = Bounds_p(1,D_Bounds_Check);
         
                % Finding temporary cost of the particle
                [~,~,Temp_cost] = cost_function(Particle_s1(i).position,number_of_variables,Func);
                if(S1_best_it(count) > Temp_cost)
                    S1_best_it(count) = Temp_cost;
                end
                if(mod(count,M)==0)
                    if(S1_worst.value<= Temp_cost)
                        S1_worst.value = Temp_cost;
                        S1_worst.member = i;
                    end
                    if(S1_best.value >= Temp_cost)
                        S1_best.value = Temp_cost;
                        S1_best.member = i;
                    end
                end
                % If temporary cost is less than particle best cost
                if Temp_cost < Particle_s1(i).best_cost
                    Particle_s1(i).best_cost = Temp_cost;
                    Particle_s1(i).best_position = Particle_s1(i).position;
                end
                % If global best cost is greater than particle best cost 
                if Particle_s1(i).best_cost < Global_s1.best_cost
                    Global_s1.best_cost = Particle_s1(i).best_cost;
                    Global_s1.best_position = Particle_s1(i).best_position;
                end 
            end
            if(count <= M)
                for j = 1:count-1
                    Fl_s1(count) = Fl_s1(count)+(S1_best_it(count-j)-S1_best_it(count-j+1))/abs(S1_best_it(count-j));
                end
            else
                for j = 1:M
                    Fl_s1(count) = Fl_s1(count)+(S1_best_it(count-j)-S1_best_it(count-j+1))/abs(S1_best_it(count-j));
                end
            end
        end
        if(Extinction_s2 ~= 1)
            gamma = 1/6+rand(1,1)*(1/2-1/6);
            ck = 20/(count)^gamma;
            for i = 1:q2
                gradient = zeros(1,number_of_variables);
                e = eye(number_of_variables);
                for j = 1:number_of_variables
                    [~,~,Temp_cost1] = cost_function(Particle_s2(i).position+ck*e(j,:),number_of_variables,Func);
                    [~,~,Temp_cost2] = cost_function(Particle_s2(i).position-ck*e(j,:),number_of_variables,Func);
                    gradient(j) = (Temp_cost1-Temp_cost2)/(2*ck);
                end
                Particle_s2(i).prevposition = Particle_s2(i).position;
                Particle_S2(i).position = Particle_s2(i).position-hk*gradient;
                hk = norm(Particle_s2(i).position-Particle_s2(i).prevposition)/norm(gradient);
                % If particle position exceeds bounds it is set to bounds
                U_Bounds_Check = find(Particle_s2(i).position > Bounds_p(2,:));
                D_Bounds_Check = find(Particle_s2(i).position < Bounds_p(1,:));
                Particle_s2(i).position(U_Bounds_Check) = Bounds_p(2,U_Bounds_Check);
                Particle_s2(i).position(D_Bounds_Check) = Bounds_p(1,D_Bounds_Check);
         
                % Finding temporary cost of the particle
                [~,~,Temp_cost] = cost_function(Particle_s2(i).position,number_of_variables,Func);
                if(S2_best_it(count) > Temp_cost)
                    S2_best_it(count) = Temp_cost;
                end
                if(mod(count,M)==0)
                    if(S2_worst.value <= Temp_cost)
                        S2_worst.value = Temp_cost;
                        S2_worst.member = i;
                    end
                    if(S2_best.value >= Temp_cost)
                        S2_best.value = Temp_cost;
                        S2_best.member = i;
                    end
                end
                % If temporary cost is less than particle best cost
                if Temp_cost < Particle_s2(i).best_cost
                    Particle_s2(i).best_cost = Temp_cost;
                    Particle_s2(i).best_position = Particle_s2(i).position;
                end
                % If global best cost is greater than particle best cost 
                if Particle_s2(i).best_cost < Global_s2.best_cost
                    Global_s2.best_cost = Particle_s2(i).best_cost;
                    Global_s2.best_position = Particle_s2(i).best_position;
                end 
            end
            if(count <= M)
                for j = 1:count-1
                    Fl_s2(count) = Fl_s2(count)+(S2_best_it(count-j)-S2_best_it(count-j+1))/abs(S2_best_it(count-j));
                end
            else
                for j = 1:M
                    Fl_s2(count) = Fl_s2(count)+(S2_best_it(count-j)-S2_best_it(count-j+1))/abs(S2_best_it(count-j));
                end
            end
        end   
        if((mod(count,M) == 0)&&(Extinction_s1 == 0)&&(Extinction_s2 == 0))
            Dp_s1_prime = Dp_s1(count-1) +Fl_s1(count)/M;
            Dp_s2_prime = Dp_s2(count-1) +Fl_s2(count)/M;
            Dp_s1(count) = Dp_s1_prime /(Dp_s1_prime +Dp_s2_prime);
            Dp_s2(count) = Dp_s2_prime /(Dp_s1_prime +Dp_s2_prime);
            if(rand <Dp_s1(count))
                q1 = q1+1;
                q2 = q2-1;
                %s2_worst = S2_worst.member
                %s1_best = S1_best.member
                Particle_s2(S2_worst.member)= [];
                Particle_s1(q1).position = Particle_s1(S1_best.member).position+.1*rand(1,number_of_variables).*(Bounds_p(2,:)-Bounds_p(1,:));
                Particle_s1(q1).velocity = zeros(1,number_of_variables);
                Particle_s1(q1).best_position = Particle_s1(q1).position;
                [~,~,Particle_s1(q1).best_cost]= cost_function(Particle_s1(q1).position,number_of_variables,Func); 
                S2_worst.value = -inf;S1_worst.value = -inf;S1_best.value = inf;S2_best.value = inf;
            else
                q1 = q1-1;
                q2 = q2+1;
                %s1_worst = S1_worst.member
                %s2_best =S2_best.member
                Particle_s1(S1_worst.member)= [];
                Particle_s2(q2).position = Particle_s2(S2_best.member).position+.1*rand(1,number_of_variables).*(Bounds_p(2,:)-Bounds_p(1,:));
                Particle_s2(q2).prevposition = zeros(1,number_of_variables);
                Particle_s2(q2).best_position = Particle_s2(q2).position;
                [~,~,Particle_s2(q2).best_cost]= cost_function(Particle_s2(q2).position,number_of_variables,Func); 
                S2_worst.value = -inf;S1_worst.value = -inf;S1_best.value = inf;S2_best.value = inf;
            end
            if(q1 <= 2) 
                Extinction_s1 = 1;
            end
            if(q2 <= 2)
                Extinction_s2 = 1;
            end
        else
            Dp_s1(count) = Dp_s1(count-1);
            Dp_s2(count) = Dp_s2(count-1);
        end
        
        
        count = count +1;    
        
    end
    G1 = Global_s1.best_cost;
    G2 = Global_s2.best_cost;
    
    Cluster_num = 2; % number of clusters at the start of the algorithm
    Desired_cluster = 2; % desired number of clusters
    Initial_clusters = randperm(q1,Cluster_num); % random index in population to save particle positions as initial cluster centers
    Cluster_min_memb = 1; % minimum members in a cluster
    Cluster_max_std = .01; % cluster maximum standard deviation for splitting
    Cluster_min_dis = 1; % cluster minimum distance for merging
    Cluster = cell(Cluster_num,2);
    for i = 1:Cluster_num
        Cluster{i,1} = Particle_s1(Initial_clusters(i)).position; % Initialize cluster centers as random particle positions
    end
    % Calling ISODATA Algorithm
    Cluster =ISODATA_new(Particle_s1,Cluster,Cluster_num,Cluster_min_memb,Cluster_max_std,Cluster_min_dis,number_of_variables,Desired_cluster);     
    nc = length(Cluster(:,1));
end
    
    
    
    