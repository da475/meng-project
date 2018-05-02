function [X_0, X_0_eval] = Best_points(Cluster,Particle,number_of_variables) 
    X_0 = [];
	X_0_eval = [];
    for i = 1: length(Cluster(:,1)) % for all cluster centers in cluster
        [temp, temp_val] = Best_calc(Cluster(i,:),Particle,number_of_variables); % return best points
        X_0 = [X_0;temp]; %add to total best points
		X_0_eval = [X_0_eval;temp_val'];
    end
        
   function [Best_pos, Best_val] = Best_calc(Cluster,Particle,number_of_variables)
        Index_Total = cell2mat(Cluster(1,2)); % all of members in cluster center are converted from cell to matrix
        Best_val = inf+zeros(1,3); % best value is inf
        Best_pos = zeros(3,number_of_variables); % best position is 0
        for i = 1:length(Index_Total) % for all members        
			cmd = num2str(Particle(Index_Total(i)).position);
			cmd = ['python svm_model.py ' cmd];

			[~, result] = system(cmd);
            Temp_best(i) = str2num(result);

            large = find(Best_val == max(Best_val)); % index of largest best_val 
            if(Best_val(large(1))> Temp_best(i)) % if largest best val is larger then current function eval 
                Best_val(large(1)) = Temp_best(i); % largest best val is current function eval
                Best_pos(large(1),:) = Particle(Index_Total(i)).position; % best position of large index is best position
            end
            
        end
        index = find(Best_val == inf); % if index is never changed that is there are less than 3 members of cluster delete best position
        if(isempty(index)== 0)
            Best_pos(index,:) = [];
			Best_val(1, index) = [];
        end
        Best_pos = [Best_pos; cell2mat(Cluster(1,1))]; % add cluster center to best position points

		cmd = num2str(cell2mat(Cluster(1, 1)));
		cmd = ['python svm_model.py ' cmd];

		[~, result] = system(cmd);
		Best_val = [Best_val str2num(result)];
   end     
end
