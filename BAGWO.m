%____________________________________________________________________________________
% Multiple strategy enhanced hybrid algorithm BAGWO combining beetle antennae search and grey wolf optimizer for global optimization
% (BAGWO)source codes version 1.2
%
% Update Date: 2025/03/15
%
% Developed in MATLAB R2022b
%
% Author and programmer: Fan Zhang
% Main Contributors: Fan Zhang, Chuankai Liu, Peng Liu, Shuiting Ding
%
% E-Mail: fan.zhang@buaa.edu.cn, liuchuankai@buaa.edu.cn, auroraus2020@outlook.com
% Project homepage: https://github.com/auroraua/BAGWO
%____________________________________________________________________________________

% BAGWO
%_________________________________________________
% Input Parameter
% N: Number of search agents in the swarm
% Max_iteration: Maximum number of iteration times
% lb: Lower limit vector of decision variable
% ub: Upper limit vector of decision variable
% dim: Number of decision variables
% fobj: The objective function corresponding to the optimization problem
%_________________________________________________
function [bestFitness,bestX,convergenceCurve]=BAGWO(N,Max_iteration,lb,ub,dim,fobj)

% Decision variable vector upper and lower limit processing
if(max(size(ub)) == 1)
   ub = ub.*ones(1,dim);
   lb = lb.*ones(1,dim);  
end
convergenceCurve = [];
xbd = ub-lb;  % The difference vector between upper limit and lower limit of decision variables

% BAGWO Initial parameter setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c = 1.0;               % Initial antennae length of beetles
localStep = 10;        % The frequency of local exploitation
finalCharisma = 0.99;  % The final charisma
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

charisma = 0;   % Initial charisma

% Latin hypercube sampling of initial decision variables of the swarm, or alternatively, random uniform sampling may be selected.
X= lhsdesign(N, dim).*xbd+lb;  

tempX = ones(N,dim);     % Used to update and record the search agent's individual best case in local exploitation

Fitness = ones(1,N).*1e50;  % Initialize swarm fitness
bestX = X(1,:);             % The decision variables vector corresponding to the historical best search agent 
bestFitness = fobj(bestX);  % The fitness corresponding to the historical best search agent 

a = 1/Max_iteration^1;     % anterior antennae length coefficient, Equation(13)
Ns = ceil(Max_iteration*((0.5)^(0.6342*Max_iteration^0.1775)));  %The number of iterations corresponding to the change in the decay rate of the beetles antennae length, Equation(16)
decayRate = (a/c)^(1/Ns);      % Decay rate of beetles length of antennae, contained within Equation(11)

% Iterative solution of optimization problem
for iterNum = 1:Max_iteration 
    for k = 1:N
        % Local exploitation process for each search agent
        for m = 1:ceil(localStep*cos(pi/2*iterNum/Max_iteration))  % Equation(16)
            dir = unifrnd(-1,1,[1,dim]);    % The orientation of the beetle's head is random
            dir = dir./norm(dir);           % Normalized, Equation(17)
            arm = dir.*c.*xbd;              % Antenna detection distance
            xr = X(k,:) + arm;              % Update the position of the right antennae, Equation(18)
            xr = max(xr,lb);xr=min(xr,ub);  % Ensure that the position of the right antennae corresponding to the right antennae are within the range of the upper limit and lower limit
            xl = X(k,:)-arm;
            xl = max(xl,lb);xl=min(xl,ub);  
            fr = fobj(xr);                  % Get the corresponding fitness at the right antennae
            fl = fobj(xl);
            minv = min(fr,fl);

            % Update search agent individual best record 
            if minv <Fitness(k)           
                Fitness(k) = minv;      % Equation(19)
                % Update and record the current search agent's best search in local exploitation
                if fr<fl
                    tempX(k,:)=xr;
                else
                    tempX(k,:)=xl;
                end
                X(k,:) = X(k,:)-2.0.*arm.*sign(fr-fl);        % Update the current search agent's location, Equation(20)
                X(k,:) = max(X(k,:),lb);X(k,:)=min(X(k,:),ub);
            else
                X(k,:) = X(k,:)-0.5.*arm.*sign(fr-fl);        % Update the current search agent's location, Equation(21)
                X(k,:) = max(X(k,:),lb);X(k,:)=min(X(k,:),ub);
            end
        end
    end

    % Update historical best search agent, Equation(6) and (7)
    indexOfSort = [];  %Record the index of search agents in the swarm, sorted by fitness
    fitnessSort = sort(Fitness);
    for index =1:N
        tempIx = find(Fitness == fitnessSort(index));
        indexOfSort(end+1) = tempIx(1);
    end
    if iterNum == 1
        bestFitness = Fitness(indexOfSort(1));
        bestX = tempX(indexOfSort(1),:);

    else
        if Fitness(indexOfSort(1)) < bestFitness
            bestFitness = Fitness(indexOfSort(1));
            bestX = tempX(indexOfSort(1),:);
        end
    end

    convergenceCurve(end+1) = bestFitness;  

    % Summon search agents in the swarm
    for pop = 1:N
        X(pop,:) = X(pop,:)+charisma.*(bestX-X(pop,:));    % Equation(22)
    end

    % 
    if iterNum == Ns
        b = 10^(-1*(0.7928*Max_iteration^0.5031));    % Calculate hind antennae length coefficient b by Equation (14)
        decayRate = (b/c)^(1/(Max_iteration-Ns));     % Update antennae length decay rate, contained within Equation (11).
    end

    charisma = 1/(1+100*((1-finalCharisma)/100)^(iterNum/Max_iteration));  % Update the charisma, Equation(8)
    c = c*decayRate;      % Update antennae length by Equation (11)
end
end
