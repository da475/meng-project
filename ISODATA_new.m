function Cluster = ISODATA_new(Particle,Cluster,Cluster_num,Cluster_min_memb,Cluster_max_std,Cluster_min_dis,number_of_variables,Desired_cluster)
    
    % Definitions
    K0 = Desired_cluster; % K0 is desired number of clusters
    K = Cluster_num; % K is acutal number of clusters
    
    % Assigning members to clusters
    Cluster1 = Cluster_Assign(Particle,Cluster,K);
    % Deleting empty clusters or cluster that have too few members
    [Cluster2, Free_points, K] = Cluster_Delete(Cluster1,Cluster_min_memb,K);
    
    % Re-adding freed members to clusters
    if (length(Free_points)~=0)
        Cluster3 = Re_Cluster(Particle,Cluster2,Free_points,K);
    else
        Cluster3 = Cluster2;
    end
    % Calculating cluster centers
    Cluster4 = Cluster_Center(Particle,Cluster3,number_of_variables,K);
   
    % Checking for Cluster Dispersion
    if K <= K0/2 % Cluster Splitting
        [Cluster5,K] = Cluster_Spliting(Particle,Cluster4,number_of_variables,Cluster_max_std,Cluster_min_memb,K);
        Cluster = Cluster5;
    elseif (K > K0*2) % Cluster Merging
        [Cluster5,K] = Cluster_Merging(Cluster4,Cluster_min_dis,K);
        Cluster = Cluster5;
    else 
        Cluster = Cluster4; % No need for merging or splitting
        Cluster5 = {};
    end
    % Empty second clumn of cluster cell that contains members belonging to cluster
    for i = 1:length(Cluster(:,1))
            Cluster{i,2} = [];
    end
    % Re-assigning members to clusters
    Cluster1 = Cluster_Assign(Particle,Cluster,K);
    Cluster = Cluster1;
     
    
% END OF FUNCTION 

% FUNCTION CALLS TO THE SUB FUNCTIONS
    
    
    % ASSIGNING PARTICLES TO CLUSTERS
    function  Cluster = Cluster_Assign(Particle,Cluster,K)
        Particle_num = length(Particle); % Number of Particles
        for i = 1:Particle_num
            Dist = zeros(1,K); % Distance matrix initialized to 0 for number of clusters
            for j = 1:K
                Dist(1,j) = norm([Particle(i).position-Cluster{j,1}]); %Distance for a particle to each cluster center
            end
            [Dist_min,Index] = min(Dist); % Finding smallest Distance, Index corresponds to the cluster center 
            Cluster{Index,2} = [Cluster{Index,2} i]; % Adding the particle to that cluster center by adding to second column of cell array at correct index
        end
    end

    % CLUSTER DELETE
    function  [Cluster,Free_particles, K] = Cluster_Delete(Cluster,Cluster_min_memb,K)
        Free_particles = []; % Initialize free particle matrix at the start of function
        i = 1;
        while (i <= K) % For each cluster
            Cluster_current_memb = length(Cluster{i,2}); % Obtaining number of members of each cluster
            if (Cluster_current_memb < Cluster_min_memb) && (Cluster_current_memb ~= 0) % If cluster has too few members and is not empty
                Free_particles = [Free_particles Cluster{i,2}]; % Add members to the free particle matrix
                Cluster(i,:) = []; % Delete the custer cell row
                K = K-1; % Decrease number of clusters
            elseif (Cluster_current_memb == 0) % If cluster center has no members
                Cluster(i,:) = []; % Delete cluster cell row
                K = K-1; % Reduce number of clusters
            else % If cluster center needs no changes go to the next
                i = i+1; 
            end
        end
    end

    % RE-CLUSTERING MEMBERS OF DELETED CLUSTER CENTERS
    function  Cluster = Re_Cluster(Particle,Cluster,Free_points,K)
        Particle_num = length(Free_points); % number of particles that need to be re-added
        for i = 1:Particle_num 
            Dist = zeros(1,K);
            for j = 1:K
                Dist(1,j) = norm([Particle(Free_points(i)).position-Cluster{j,1}]); % calculating distance from each unassigned particle to each cluster center
            end
            [Dist_min,Index] = min(Dist);
            Cluster{Index,2} = [Cluster{Index,2} Free_points(i)];
        end
    end
    
    % OBTAINING NEW CLUSTER CENTERS
    function Cluster = Cluster_Center(Particle,Cluster,number_of_variables,K)
        for i = 1:K % for each cluster
            sums = zeros(1,number_of_variables); % sum is a zeros matrix of length number of variables
            temp = cell2mat(Cluster(i,2)); % temp is the members belonging to cluster converted from cell to matrix type
            num_members = length(Cluster{i,2}); % number of members belonging to each cluster
            for j = 1:num_members
                sums = sums + Particle(temp(j)).position; % Adding particle poisitons of all members in a cluster
            end
            Cluster{i,1} = (1/num_members)*sums; % devide the sum by the number of members in cluster to find the average position
        end
    end
    
    % SPLITTING CLUSTERS
    function [Cluster,K] = Cluster_Spliting(Particle,Cluster,number_of_variables,Cluster_max_std,Cluster_min_memb,K)
        % Calculating standard deviation of each cluster (STANDARD DEVIATION FORMULA)
        for i = 1:K
            sumation = zeros(1,number_of_variables); % sumation part of standard deviation formula
            temp = cell2mat(Cluster(i,2)); % temp is the members belonging to cluster converted from cell to matrix type
            num_members = length(Cluster{i,2}); % number of members belonging to each cluster
            for j = 1:num_members % obtaining sumation
                sumation = sumation + (Particle(temp(j)).position-cell2mat(Cluster(i,1))).^2;
            end
            Std{i,1} = sqrt((1/num_members)*sumation); % Computing standard deviation of each variable in Cluster in first column of cell array
            Std{i,2} = max(cell2mat(Std(i,1)));% Save max std of varaible of current cluster evaluated in second column of cell array
            temp = []; % deleting contents of temp
        end    
        
        m = 1; % m is a counter
        temp_K = K; % temp K is another counter index for std cell
        while (m <= temp_K)
            num_members = length(Cluster{m,2}); % obtain number of variables belonging to a cluster
            if (Std{m,2} > Cluster_max_std) && (num_members> Cluster_min_memb*2) % If standard deviation is too big and there are enough members to split
                temp = zeros(1,number_of_variables); % matrix of zeros with size equal to number of variables
                Location = find(Std{m,1} == Std{m,2}); % Location is the index location of largest std in in cluster position std 
                temp(Location) = temp(Location)+.0001*Std{m,2}; % displacing the variable with some distance
                Cluster(end+1,:) = {Cluster{m,1}-temp,[]}; % adding a new cluster center by subtracting distance offset from current center
                Cluster(end+1,:) = {Cluster{m,1}+temp,[]}; % adding a new cluster center by adding distance offset from current center
                Cluster(m,:) = []; % Deleting current center evaulated
                Std(m,:) = []; % deleting std matrix of current center
                K = K+1; % changing number of cluster to include the new clusters
                temp_K = temp_K-1; % since std cell row is deleted, next std to be evaluated has same index
           else 
               m = m+1; % if cluster is not split then next row of std is evaluated
           end    
       end
    end
    
    % CLUSTER MERGING
    function [Cluster,K] = Cluster_Merging(Cluster,Cluster_min_dis,K)
        Distance = [];
        Lump_clus = [];
        % 
        if (isempty(Cluster)) % if cluster is empty do nothing (not a necessary check but prevent program from breaking)
            return;
        end
        % creating a matrix that is an adjaceny list of distances betwween
        % clsuters
        for i = 1:K
           for j = i+1:K
                 Distance = [Distance; i,j,norm(Cluster{i,1}-Cluster{j,1})];
            end
        end
        if(isempty(Distance) || (length(Distance(:,1)) == 1)) % if distance is empty (only one cluster) or has one vector (only two clusters) dont merge
            return;
        end
        Sorted_Distance = sortrows(Distance,3) % Sort Distance based on distance 
        N = find(Sorted_Distance(:,3) < Cluster_min_dis) % Find index of all of the sorted distances that are too close to each other
        if(isempty(N)== 0) % Return if not clusters need merging
            return;
        end
        for m = 1:length(N) % Merge clusters
            if (ismember(Sorted_Distance(N(m),1),Lump_clus) == 0) && (ismember(Sorted_Distance(N(m),2),Lump_clus) == 0) % if the two clusters have not been combined yet
                Lump_clus(end+1) = Sorted_Distance(N(m),1); % add clusters to the clusters that have been combined
                Lump_clus(end+1) = Sorted_Distance(N(m),2);
                Cluster{Sorted_Distance(N(m),1),1} = (Cluster{Sorted_Distance(N(m),1),1}+Cluster{Sorted_Distance(N(m),2),1})/2; % Editing current cluster 
                Cluster{Sorted_Distance(N(m),2),1} = []; % Deleting other 
                K = K-1; % decreasing number of clusters
            end
        end
        % go though each cluster in cell array and delete those who have
        % been merged but not modified
        n = length(Cluster(:,1));
        while n ~= 0
            if isempty(Cluster{n,1})
                Cluster(n,:) = [];
            end
            n= n-1;
        end
        
   end
end