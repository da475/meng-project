
function [lower_bound, upper_bound, cost] = cost_function(particle,num_variables,function_choice)
    switch function_choice
        case 1
            % Rastrigin Function
            cost = 10*num_variables+sum(particle.^2-10*cos(2*pi.*particle));
            lower_bound = -5.12;
            upper_bound = 5.12;
        case 2
            % Shifted Sphere Function
            cost = sum(particle.^2)-450;
            lower_bound = -100;
            upper_bound = 100;
        case 3
            % Griewank Function
            Product = cos(particle(1)/sqrt(1));
            for i = 2:num_variables
                Product = Product*cos(particle(i)/sqrt(i));
            end
            cost = 1+(1/4000)*sum(particle.^2)-Product;
            lower_bound = -600;
            upper_bound = 600;
        case 4
            % Shifted Rosenbrock
            cost = 0;
            for i = 1:num_variables-1
                cost = cost+100*(particle(i+1)-particle(i)^2)^2+(1-particle(i))^2;
            end
            cost = cost+390;
            lower_bound = -100;
            upper_bound = 100;
        case 5
            % Shifted Rotated Ackley
            cost = -20*exp(-.02*sqrt(sum((particle.^2)/num_variables)))-exp(sum(cos(2*pi.*particle))/num_variables)+20+exp(1)+(-140);
            lower_bound = -32;
            upper_bound = 32;
        otherwise
            disp('Cost Unavailable')
    end
    
            
end



    