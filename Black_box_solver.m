function [xb,xb_eval] = Black_box_solver(x0,N,Func)
    
    n=N+1; %value of N+1
    alpha=1; %reflection coefficient
    gamma=2; %expansion coefficient
    rho=0.5; %contraction coefficient
    sigma=0.5; %shrink coefficient
        
    for j = 1:length(x0(:,1))
        Vertices = CreateInitialPolytope(x0(j,:),n,Func);
        StdVal=10; % Initial standard deviation of that will not converge while loop
        while((StdVal >= 1*10^-15))
            Vertices = Sort(Vertices,n); % Sorting verticies
            Centroid = Calculate_Centroid(Vertices,n,Func); % Calculating Centroid
            StdVal = Calculate_STD(Vertices,n); % Obtaining standard deviation
  
            % Reflection coordinate
            Reflect.coord = Centroid.coord +alpha.*(Centroid.coord-Vertices(n).coord); 
            Reflect.value = f(Reflect.coord,n,Func);
            % Perform Reflection
            if((Vertices(1).value <= Reflect.value) && (Reflect.value < Vertices(n-1).value))
                Vertices(n)=Reflect;
                continue; 
            end
            % Perform Expansion
            if(Reflect.value < Vertices(1).value) 
                Expand.coord = Centroid.coord+gamma.*(Reflect.coord-Centroid.coord);
                Expand.value = f(Expand.coord,n,Func);
        
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
                ContractOut.value = f(ContractOut.coord,n,Func);
                if(ContractOut.value <= Reflect.value)
                    Vertices(n) = ContractOut;
                    continue;
                end
            % Perform Inside Contraction
            elseif(Reflect.value >= Vertices(n).value)
                ContractIn.coord = Centroid.coord - rho.*(Reflect.coord-Centroid.coord);  %Contract Inside
                ContractIn.value= f(ContractIn.coord,n,Func);
                if(ContractIn.value <Vertices(n).value)
                    Vertices(n) = ContractIn;
                    continue
                end
            end
            % Perform Shrink
            for i=2:n
                  Vertices(i).coord = Vertices(1).coord + sigma.*(Vertices(i).coord-Vertices(1).coord);
                  Vertices(i).value = f(Vertices(i).coord,n,Func);   
            end
        end
    Minima=Vertices(1);
    xb(j,:) = Minima.coord;
    xb_eval(j,1) = Minima.value;
    end
end
% FUNCTIONS EVALUATION CALLS
function [cost]=f(V,n,Func) %Write your function in matrix form
     num_variables = n-1;
      switch Func
        case 1
            % Rastrigin Function
            cost = 10*num_variables+sum(V.^2-10*cos(2*pi.*V));
            %lower_bound = -5.12;
            %upper_bound = 5.12;
        case 2
            % Shifted Sphere Function
            cost = sum(V.^2)-450;
           % lower_bound = -100;
           % upper_bound = 100;
        case 3
            % Griewank Function
            Product = cos(V(1)/sqrt(1));
            for i = 2:num_variables
                Product = Product*cos(V(i)/sqrt(i));
            end
            cost = 1+(1/4000)*sum(V.^2)-Product;
           % lower_bound = -600;
           % upper_bound = 600;
        case 4
            % Shifted Rosenbrock
            cost = 0;
            for i = 1:num_variables-1
                cost = cost+100*(V(i+1)-V(i)^2)^2+(1-V(i))^2;
            end
            cost = cost+390;
            %lower_bound = -100;
            %upper_bound = 100;
        case 5
            % Shifted Rotated Ackley
            cost = -20*exp(-.02*sqrt(sum((V.^2)/num_variables)))-exp(sum(cos(2*pi.*V))/num_variables)+20+exp(1)+(-140);
            %lower_bound = -32;
            %upper_bound = 32;
        otherwise
            disp('Cost Unavailable')
      end
    
end
% CREATING POLYTOPE
function Vertices = CreateInitialPolytope(x0,n,Func)
    % First Vertex is Initial point
    Vertices(1).coord=x0; 
    Vertices(1).value=f(Vertices(1).coord,n,Func);
    % random generation of other veritices
    for i=2:n
        Vertices(i).coord=Vertices(1).coord+.01.*rand(1,(n-1)); 
        Vertices(i).value=f(Vertices(i).coord,n,Func);
    end
    
end
% SORTING VERTICES
function [SortVertices]=Sort(Vertices,n)
    % sorting vertices from smallest to largest
    [~,Order] = sort([Vertices.value]);
    SortVertices = Vertices(Order);
end
% CALCULATING CENTROID OF THE POLYTOPE
function [Centroid]=Calculate_Centroid(Vertices,n,Func)
    % Obtaining Average Centroid
    Sum=zeros(1,(n-1));
    for i=1:n-1
        Sum=Sum+Vertices(i).coord;
    end
    Centroid.coord=Sum./(n-1);
    Centroid.value=f(Centroid.coord,n,Func);
end
% CALCULATING STANDARD DEVIATION OF VERTICES
function[StdVal]=Calculate_STD(Vertices,n) % this is the tolerance value, the standard deviation of the converging values
    for i=1:n
        ValueArray(i)=Vertices(i).value;
    end
    StdVal=std(ValueArray,1);
end
