clear; clc; close all;

% __________________________________________________________
%Part b starts
%___________________________________________________________

% Load network topology
run('topology.m');

Num_Links=12;
Num_Flows=7;
Max_Links_On_Path=4;

Link_Capacity=[1.0 1.0 1.0 1.0 1.0 1.0 2.0 2.0 2.0 2.0 2.0 2.0];
Flow_Weight=[1.0 1.0 1.0 1.0 2.0 2.0 2.0];

Flow_Path=[ [3 9 -1 -1]; ...
    [4 9 -1 -1]; ...
    [3 10 -1 -1]; ...
    [4 10 -1 -1]; ...
    [5 8 -1 -1]; ...
    [5 9 -1 -1]; ...
    [1 6 11 -1]];

% Create an Incidence matrix A: Num_Links x Num_Flows
A = zeros(Num_Links, Num_Flows);
for i = 1:Num_Flows
    for link = Flow_Path(i, :)
        if link > 0
            A(link, i) = 1;
        end
    end
end
% Initialization
lambda = zeros(Num_Links, 1);
alpha = 0.01;
max_iters = 4000;
flow_rates_history = zeros(max_iters, Num_Flows);
lambda_history = zeros(max_iters, Num_Links);
% Dual Gradient Ascent Loop
for it = 1:max_iters
    % Compute flow rates
    x = zeros(Num_Flows, 1);
    for i = 1:Num_Flows
        links = find(A(:, i));
        total_lambda = sum(lambda(links));
        if total_lambda > 0
            x(i) = Flow_Weight(i) / total_lambda;
        else
            x(i) = 10;
        end
    end
    flow_rates_history(it, :) = x';
    lambda_history(it, :) = lambda';
    % Update dual variables
    lambda = max(lambda + alpha * (A * x - Link_Capacity'), 0);
end
final_flows = flow_rates_history(end, :)'
% Plot Flow Rate Convergence
figure;
plot(flow_rates_history, 'LineWidth', 1.5);
xlabel('Iterations');
ylabel('Flow Rates');
title('Convergence of Flow Rates');
legend(arrayfun(@(i) sprintf('Flow %d', i), 1:Num_Flows, 'UniformOutput', false));
grid on;
% Plot Lambda Convergence
figure;
plot(lambda_history, 'LineWidth', 2);
xlabel('Iterations');
ylabel('Lambda Values λ');
title('Dual Variable Convergence (Lambda λ) ');
legend(arrayfun(@(i) sprintf('\\lambda_{%d}', i), 1:Num_Links, 'UniformOutput', false));
%applies the function func to the elements of A, 
% one element at a time. 
% arrayfun then concatenates the outputs from func into the output array B,
% so that for the ith element of A, B(i) = func(A(i)).
grid on;

% __________________________________________________________
%Part b ends
%___________________________________________________________



% __________________________________________________________
%Part c starts
%___________________________________________________________

% Optimality Check
% 1. Primal Feasibility: Ax <= c
Ax = A * final_flows;
primal_residual = Ax - Link_Capacity(:);
fprintf('Max constraint violation (Ax <= c): %.2e\n', max(primal_residual));
% 2. Dual Feasibility:
fprintf('Min dual variable: %.2e\n', min(lambda));
% 3. Complementary Slackness: λᵗ(Ax - c)
comp_slackness = lambda' * (Ax - Link_Capacity(:));
fprintf('Complementary slackness : %.2e\n', comp_slackness);
% 4. Stationarity:
w = Flow_Weight(:);
stationarity_error = zeros(Num_Flows, 1);
for i = 1:Num_Flows
    links = find(A(:, i));
    total_lambda = sum(lambda(links));
    if total_lambda > 0
        x_check = w(i) / total_lambda;
    else
        x_check = 10;
    end
    stationarity_error(i) = abs(final_flows(i) - x_check);
end
fprintf('Max stationarity error: %.2e\n', max(stationarity_error));

%Checking KKT Conditions with thresholds

if max(primal_residual) < 1e-3 && ...
   min(lambda) > -1e-6 && ...
   abs(comp_slackness) < 1e-3 && ...
   max(stationarity_error) < 1e-6
   disp('KKT conditions satisfied - The given solution is optimal');
else
   disp('Solution does not satisfy KKT conditions');
end
 

% __________________________________________________________
%Part c ends
%___________________________________________________________