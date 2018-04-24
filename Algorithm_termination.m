% Test for ISO DATA convergence
function Result = Algorithm_termination(Test1,Test2,Test3)
    % Result of 0 means that there is no convergence 
    Result = 0;
    % If one of the clusters is empty then convergence can not happen
    if(isempty(Test1)||isempty(Test2)||isempty(Test3))
        return;
    end
    % Check for whether the three clusters have the same number of centers,
    % if not, there is no convergence
    Num_of_clusters = [length(Test1(:,1)) length(Test2(:,1)) length(Test3(:,1))];
    if(range(Num_of_clusters) ~= 0)
        return;
    end
    
    Clust_mem = zeros(3,Num_of_clusters(1)); % matrix with rows being different clusters, and columns the number of members in each cluster cented
    Clust_mem(1,:) = sort(Cluster_members(Test1,Num_of_clusters(1)));
    Clust_mem(2,:) = sort(Cluster_members(Test2,Num_of_clusters(2)));
    Clust_mem(3,:) = sort(Cluster_members(Test3,Num_of_clusters(3)));
    
    if(isequal(Clust_mem(1,:),Clust_mem(2,:)) == 0) % there are a different number of members in each cluster array then no convergence
        return;
    end
    if(isequal(Clust_mem(1,:),Clust_mem(3,:)) == 0)
        return;
    end
    result = zeros(1,2); % checking if the individual members are the same
    result(1,1) = check_member(Test1,Test2,Num_of_clusters(1));
    result(1,2) = check_member(Test1,Test3,Num_of_clusters(1));
    if(sum(result) ~= 2) % if not the same no convergence
        return;
    end
    % if everything is same convergence is reached return 1
    Result = 1;
    
    % OBTAINING MEMBERS OF A CLUSTER FOR EACH CENTER
    function members =  Cluster_members(Test,Num_of_clusters)
        members = []; % originial matrix empty
        for i = 1:Num_of_clusters
            members = [members length(Test{i,2})]; %enter number of members of each center
        end
    end
    % COMPARING INDIVIDUAL MEMBERS OF A CLUSTER
    function result = check_member(Test1,Test2,number_of_clusters)
        result = 0;
        for i = 1:number_of_clusters % for all of the cluster centers
            temp = cell2mat(Test1(i,2)); % obtain a cluster center members from cluster in first cluster array
            for j = 1:number_of_clusters
                temp2 = cell2mat(Test2(j,2)); % obtain a cluster center members from cluster in second cluster array
                if(isequal(sort(temp),sort(temp2))) % if cluster member are same result increases 
                    result = result + 1;
                end
            end
        end
        if(result == number_of_clusters) % if all cluster members are same return 1 else return 0
            result =1;
        else
            result = 0;
        end
   end

end