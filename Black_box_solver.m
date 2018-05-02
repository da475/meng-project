function [xb,xb_eval] = Black_box_solver(x0, N)
    
    n=N+1; %value of N+1
    alpha=1; %reflection coefficient
    gamma=2; %expansion coefficient
    rho=0.5; %contraction coefficient
    sigma=0.5; %shrink coefficient
        
    for j = 1:length(x0(:,1))
        Vertices = CreateInitialPolytope(x0(j,:), n);
        StdVal=10; % Initial standard deviation of that will not converge while loop
        while((StdVal >= 1*10^-15))
            Vertices = Sort(Vertices,n); % Sorting verticies
            Centroid = Calculate_Centroid(Vertices,n); % Calculating Centroid
            StdVal = Calculate_STD(Vertices,n); % Obtaining standard deviation
  
            % Reflection coordinate
            Reflect.coord = Centroid.coord +alpha.*(Centroid.coord-Vertices(n).coord); 
			Reflect.coord = CheckBounds(Reflect.coord);
            Reflect.value = f(Reflect.coord);
            % Perform Reflection
            if((Vertices(1).value <= Reflect.value) && (Reflect.value < Vertices(n-1).value))
                Vertices(n)=Reflect;
                continue; 
            end
            % Perform Expansion
            if(Reflect.value < Vertices(1).value) 
                Expand.coord = Centroid.coord+gamma.*(Reflect.coord-Centroid.coord);
				Expand.coord = CheckBounds(Expand.coord);
                Expand.value = f(Expand.coord);
        
                if(Expand.value < Reflect.value)
                    Vertices(n) = Expand;
                    continue;
                else
                    Vertices(n) = Reflect;
                    continue;
                end 
            end
            % Perform Outside Contraction
            if((Vertices(n-1).value <= Reflect.value) && (Reflect.value < Vertices(n).value))
                ContractOut.coord = Centroid.coord + rho.*(Reflect.coord-Centroid.coord); %Contract Outside
				ContractOut.coord = CheckBounds(ContractOut.coord);
                ContractOut.value = f(ContractOut.coord);
                if(ContractOut.value <= Reflect.value)
                    Vertices(n) = ContractOut;
                    continue;
                end
            % Perform Inside Contraction
            elseif(Reflect.value >= Vertices(n).value)
                ContractIn.coord = Centroid.coord - rho.*(Reflect.coord-Centroid.coord);  %Contract Inside
				ContractIn.coord = CheckBounds(ContractIn.coord);
                ContractIn.value= f(ContractIn.coord);
                if(ContractIn.value <Vertices(n).value)
                    Vertices(n) = ContractIn;
                    continue
                end
            end
            % Perform Shrink
            for i=2:n
                  Vertices(i).coord = Vertices(1).coord + sigma.*(Vertices(i).coord-Vertices(1).coord);
				  Vertices(i).coord = CheckBounds(Vertices(i).coord);
                  Vertices(i).value = f(Vertices(i).coord);   
            end
        end
    Minima=Vertices(1);
    xb(j,:) = Minima.coord;
    xb_eval(j,1) = Minima.value;
    end
end

% FUNCTIONS EVALUATION CALLS
function cost = f(V) %Write your function in matrix form
   	cmd = num2str(V);
	cmd = ['python svm_model.py ' cmd];
	[~, result] = system(cmd);
	cost = str2num(result);
end

% CREATING POLYTOPE
function Vertices = CreateInitialPolytope(x0,n)
    % First Vertex is Initial point
    Vertices(1).coord=x0; 

	Vertices(1).value = f(Vertices(1).coord);

	Lower_Bound = [0.01, 0.1, 0.5];
	Upper_Bound = [0.5, 100, 5];
	Bounds_p = [Lower_Bound; Upper_Bound];

    % random generation of other veritices
    for i=2:n
		eps = (Bounds_p(1, :) + ((Bounds_p(2, :) - Bounds_p(1, :)).*rand(1, n-1))) / 100;

        Vertices(i).coord = Vertices(1).coord + eps;

        Vertices(i).value = f(Vertices(i).coord);
    end
    
end
% SORTING VERTICES
function [SortVertices]=Sort(Vertices,n)
    % sorting vertices from smallest to largest
    [~,Order] = sort([Vertices.value]);
    SortVertices = Vertices(Order);
end
% CALCULATING CENTROID OF THE POLYTOPE
function [Centroid]=Calculate_Centroid(Vertices,n)
    % Obtaining Average Centroid
    Sum=zeros(1,(n-1));
    for i=1:n-1
        Sum=Sum+Vertices(i).coord;
    end
    Centroid.coord=Sum./(n-1);

	Centroid.value = f(Centroid.coord);
end
% CALCULATING STANDARD DEVIATION OF VERTICES
function[StdVal]=Calculate_STD(Vertices,n) % this is the tolerance value, the standard deviation of the converging values
    for i=1:n
        ValueArray(i)=Vertices(i).value;
    end
    StdVal=std(ValueArray,1);
end

function point = CheckBounds(point)
	Lower_Bound = [0.01, 0.1, 0.5];
	Upper_Bound = [0.5, 100, 5];
	Bounds_p = [Lower_Bound; Upper_Bound];

	% If particle position exceeds bounds it is set to bounds
	U_Bounds_Check = find(point > Bounds_p(2, :));
	D_Bounds_Check = find(point < Bounds_p(1, :));
	point(U_Bounds_Check) = Bounds_p(2, U_Bounds_Check);
	point(D_Bounds_Check) = Bounds_p(1, D_Bounds_Check);
end
