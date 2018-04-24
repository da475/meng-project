function X_0 = Best_points(Cluster,Particle,number_of_variables,Func) 
    X_0 = [];
    for i = 1: length(Cluster(:,1)) % for all cluster centers in cluster
        temp = Best_calc(Cluster(i,:),Particle,number_of_variables,Func); % return best points
        X_0 = [X_0;temp]; %add to total best points
    end
        
   function Best_pos = Best_calc(Cluster,Particle,number_of_variables,Func)
        Index_Total = cell2mat(Cluster(1,2)); % all of members in cluster center are converted from cell to matrix
        Best_val = inf+zeros(1,3); % best value is inf
        Best_pos = zeros(3,number_of_variables); % best position is 0
        for i = 1:length(Index_Total) % for all members
            [~,~,Temp_best(i)]= cost_function(Particle(Index_Total(i)).position,number_of_variables,Func); % get the function eval
            large = find(Best_val == max(Best_val)); % index of largest best_val 
            if(Best_val(large(1))> Temp_best(i)) % if largest best val is larger then current function eval 
                Best_val(large(1)) = Temp_best(i); % largest best val is current function eval
                Best_pos(large(1),:) = Particle(Index_Total(i)).position; % best position of large index is best position
            end
            
        end
        index = find(Best_val == inf); % if index is never changed that is there are less than 3 members of cluster delete best position
        if(isempty(index)== 0)
            Best_pos(index,:) = [];
        end
        Best_pos(end,:) = cell2mat(Cluster(1,1)); % add cluster center to best position points
   end     
end