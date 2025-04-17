[x1, x2] = meshgrid(-5:0.1:5, -5:0.1:5); % Creating a grid of x1 and x2 values
f = f0(x1,x2); % WRITING objective function

% __________________________________________________________
%Part a starts
%___________________________________________________________

mesh(x1, x2, f0(x1,x2)); % Plot mesh surface 
title('3D Plot: f(x1, x2)');
xlabel('x1 value');
ylabel('x2 value');
zlabel('f(x1, x2)');

% Randomly chosingg directions
directions = [1, 1; -1, 1; 2, -1]; 
x1_start = 0;
x2_start = 0;
figure;
hold on;
for i = 1:size(directions, 1)
    % Parametrize the line: x = x_start + t * direction
    t = linspace(-5, 5, 100);
    x1_line = x1_start + t * directions(i, 1);
    x2_line = x2_start + t * directions(i, 2);
    
    % Compute f along the line
    f_line = x1_line.^2 + 3*x1_line.*x2_line + 9*x2_line.^2 + 2*x1_line - 5*x2_line;
    plot(t, f_line);
end
hold off;
mycolors = [0 0 1; 0 1 0; 1 0 1];
ax = gca; 
ax.ColorOrder = mycolors;
xlabel('t');
ylabel('f(t)');
title('Function Restricted to Lines');
legend('Direction 1', 'Direction 2', 'Direction 3');



% __________________________________________________________
%Part a ends
%___________________________________________________________

% __________________________________________________________
%Part b starts
%___________________________________________________________


grad_f = @(x1, x2) df0(x1,x2);

gamma = 0.1; % Step size 0.001, 0.11, and 0.1
max_iter = 1000; % max iterations required

x = [0, 0]; % Starting point [x1, x2]
x1_vals = zeros(1, max_iter);
x2_vals = zeros(1, max_iter);
for k = 1:max_iter
    grad = grad_f(x(1), x(2)); % Compute gradient
    x = x - gamma * grad; % Update the values
    
    x1_vals(k) = x(1);
    x2_vals(k) = x(2);
end
% Plotting
figure;
%subplot(2,1,1);
plot(1:max_iter, x1_vals, 'r', 1:max_iter, x2_vals, 'b');
legend('x1(k)', 'x2(k)');
xlabel('Iterations');
ylabel('Value');
title('Evolution of x1 and x2 with Iteration (Step size 0.1)');


%subplot(2,1,2);
figure;
contour(x1, x2, f, 50);
hold on;
plot(x1_vals, x2_vals, "k-o", 'LineWidth', 0.5, 'MarkerSize',3);
title('Contour Plot with Gradient Descent Trajectory (Step size 0.1)');
xlabel('x1'); ylabel('x2');
hold off;


% __________________________________________________________
%Part b ends
%___________________________________________________________

% __________________________________________________________
%Part c starts
%___________________________________________________________

f = f0(x1,x2);
grad_f = @(x1, x2) df0(x1,x2);
x = [5; 5]; % Starting point [x1, x2]
max_iter = 100;


ss1 = 0.001; % Small step size
[x_min1, f1, x1_history1, x2_history1] = gradient_projection(@f0, @df0, ss1, x, max_iter);

ss2 = 10.0; % Large step size
[x_min2, f2, x1_history2, x2_history2] = gradient_projection(@f0, @df0, ss2, x, max_iter);

ss3 = 0.1; % Perfect step size
[x_min3, f3, x1_history3, x2_history3] = gradient_projection(@f0, @df0, ss3, x, max_iter);


% Plot results
figure;
%subplot(3,1,1);
plot(x1_history1);
hold on;
plot(x2_history1);
title(sprintf('Small Step Size: %0.4f', ss1));
xlabel('Iterations');
ylabel('x1 and x2');
legend('x1', 'x2');

figure;
%subplot(3,1,2);
plot(x1_history2);
hold on;
plot(x2_history2);
title(sprintf('Large Step Size:%0.2f', ss2));
xlabel('Iteration');
ylabel('x1 and x2');
legend('x1', 'x2');


figure;
%subplot(3,1,3);
plot(x1_history3);
hold on;
plot(x2_history3);
title(sprintf('Perfect Step Size: %0.2f', ss3));
xlabel('Iteration');
ylabel('x1 and x2');
legend('x1', 'x2');


figure;
% Run gradient descent for each step-size
for i = 1:3
    subplot(1,3,i);
    if i == 1
        alpha = ss1;
        x1_traj = x1_history1;
        x2_traj = x2_history1;
    elseif i == 2
        alpha = ss2;
        x1_traj = x1_history2;
        x2_traj = x2_history2;
    else
        alpha = ss3;
        x1_traj = x1_history3;
        x2_traj = x2_history3;
    end
    
    % Plot the contour plot of the objective function
    [X1,X2] = meshgrid(-5:0.1:5,-5:0.1:5);
    Y = f0(X1,X2);
    contour(X1,X2,Y,20);
    xlabel('x1');
    ylabel('x2');
    hold on;
    
    % Plot the trajectory
    plot(x1_traj, x2_traj, '-o');
    title(['Step-size = ', num2str(alpha)]);
end


% Print the optimal solution and objective value
fprintf('Optimal solution: [%f, %f]\n', x_min3(1), x_min3(2));
fprintf('Objective value: %f\n', f3);


% __________________________________________________________
%Part c ends
%___________________________________________________________

% __________________________________________________________
%Part d starts
%___________________________________________________________

% Set the initial point and step size
max_iter = 1000;
lambda_init = [0;0]; 

% Small lambda
ss1 = 0.005;
[x_opt1, lambda_opt1, x_history1, lambda_history1] = dual_gradient_projection(lambda_init, ss1, max_iter);

% Large lambda
ss2 = 1.5;
[x_opt2, lambda_opt2, x_history2, lambda_history2] = dual_gradient_projection(lambda_init, ss2, max_iter);

% Perfect lambda
ss3 = 0.1;
[x_opt3, lambda_opt3, x_history3, lambda_history3] = dual_gradient_projection(lambda_init, ss3, max_iter);

% Print the optimal solution and objective value
fprintf('Optimal solution: [%f, %f]\n', x_opt3(1), x_opt3(2));
fprintf('Optimal dual variable: [%f, %f]\n', lambda_opt3(1), lambda_opt3(2));
fprintf('Objective value: %f\n', f0(x_opt3(1), x_opt3(2)));

% Plot results
figure;
subplot(3,1,1);
plot(x_history1(:, 1));
hold on;
plot(x_history1(:, 2));
hold on;
plot(lambda_history1(:, 1));
hold on;
plot(lambda_history1(:, 2));
title(sprintf('[Too small]Step Size: %0.4f', ss1));
xlabel('Iteration');
legend('x1', 'x2', 'lambda1', 'lambda2');

subplot(3,1,2);
plot(x_history2(:, 1));
hold on;
plot(x_history2(:, 2));
hold on;
plot(lambda_history2(:, 1));
hold on;
plot(lambda_history2(:, 2));
title(sprintf('[Too large]Step Size: %0.4f', ss2));
xlabel('Iteration');
legend('x1', 'x2', 'lambda1', 'lambda2');

subplot(3,1,3);
plot(x_history3(:, 1));
hold on;
plot(x_history3(:, 2));
hold on;
plot(lambda_history3(:, 1));
hold on;
plot(lambda_history3(:, 2));
title(sprintf('[Optimum]Step Size: %0.4f', ss3));
xlabel('Iteration');
legend('x1', 'x2', 'lambda1', 'lambda2');


% Plot results
figure;
subplot(3,1,1);
plot(lambda_history1(:, 1));
hold on;
plot(lambda_history1(:, 2));
title(sprintf('[Too small]Step Size: %0.4f', ss1));
xlabel('Iterations');
legend('lambda1', 'lambda2');

subplot(3,1,2);
plot(lambda_history2(:, 1));
hold on;
plot(lambda_history2(:, 2));
title(sprintf('[Too large]Step Size: %0.4f', ss2));
xlabel('Iterations');
legend('lambda1', 'lambda2');

subplot(3,1,3);
plot(lambda_history3(:, 1));
hold on;
plot(lambda_history3(:, 2));
title(sprintf('[Optimum]Step Size: %0.4f', ss3));
xlabel('Iteration');
legend('lambda1', 'lambda2');

%Figure starts
figure;
subplot(3,1,1);
plot(x_history1(:, 1));
hold on;
plot(x_history1(:, 2));
hold on;
title(sprintf('[Too small]Step Size: %0.4f', ss1));
xlabel('Iterations');
legend('x1', 'x2');

subplot(3,1,2);
plot(x_history2(:, 1));
hold on;
plot(x_history2(:, 2));
hold on;
title(sprintf('[Too large]Step Size: %0.4f', ss2));
xlabel('Iterations');
legend('x1', 'x2');

subplot(3,1,3);
plot(x_history3(:, 1));
hold on;
plot(x_history3(:, 2));
hold on;
title(sprintf('[Optimum]Step Size: %0.4f', ss3));
xlabel('Iteration');
legend('x1', 'x2');

% Figure ends


figure;
% Run gradient descent for each step-size
for i = 1:3
    subplot(1,3,i);
    if i == 1
        ss = ss1;
        x1_traj = x_history1(:, 1);
        x2_traj = x_history1(:, 2);
    elseif i == 2
        ss = ss2;
        x1_traj = x_history2(:, 1);
        x2_traj = x_history2(:, 2);
    else
        ss = ss3;
        x1_traj = x_history3(:, 1);
        x2_traj = x_history3(:, 2);
    end
    
    % Plot the contour plot of the objective function
    [X1,X2] = meshgrid(-5:0.1:5,-5:0.1:5);
    Y = f0(X1,X2);
    contour(X1,X2,Y,20);
    xlabel('x1');
    ylabel('x2');
    hold on;
    
    % Plot the trajectory
    plot(x1_traj, x2_traj, '-o');
    title(['Step-size = ', num2str(ss)]);
end


% __________________________________________________________
%Part d ends
%___________________________________________________________


% __________________________________________________________
%Part e starts
%___________________________________________________________
tol = 1e-2;

% 1.b check
fprintf('check Part 1.b \n');
x1_0 = 0;
x2_0 = 0;
max_iter = 100;

gamma3 = 0.05;
[x1_min3, x2_min3, f_min3, ~, ~] = gradient_descent(@f0, @df0, gamma3, x1_0, x2_0, max_iter);

fprintf('Optimal solution is : [%f, %f]\n', x1_min3, x2_min3);
fprintf('Objective value is : %f\n', f_min3);

gradient = df0(x1_min3, x2_min3);
fprintf('Gradient: [%f, %f]\n', gradient(1), gradient(2));
gradient_zero = all(abs(gradient) <= tol);
fprintf('Gradient of function is zero: %s\n', mat2str(gradient_zero));

% 1.c check
fprintf('check Part 1.c \n');

x1_0 = 0;
x2_0 = 0;
max_iter = 50;
x = [x1_0; x2_0];

alpha3 = 0.1;
[x_min3, f3, ~, ~] = gradient_projection(@f0, @df0, alpha3, x, max_iter);
fprintf('Optimal solution: [%f, %f]\n', x_min3(1), x_min3(2));
fprintf('Objective value: %f\n', f3);

flag_cons1 = false;
flag_cons2 = false;

if abs(2*x_min3(1)+x_min3(2)-3) <= tol
    flag_cons1 = true;
    fprintf('Constraint satisfied for constraint line: [2, -1]\n');
end

if abs(x_min3(1)+2*x_min3(2)-3) <= tol
    flag_cons2 = true;
    fprintf('Constraint satisfied for constraint line: [-2, 1]\n');
end

inner_product_zero = false;
gradient = df0(x_min3(1), x_min3(2));
fprintf('Gradient: [%f, %f]\n', gradient(1), gradient(2));

if flag_cons1
    if abs(gradient*[2;-1]) <= tol
        inner_product_zero = true;
    end
end

if flag_cons2
    if abs(gradient*[-2;1]) <= tol
        inner_product_zero = true;
    end
end

fprintf('Inner product of gradient and above constraint line is zero: %s\n', mat2str(inner_product_zero));

% Check 1.d 
fprintf('\n check Part 1.d \n');

max_iter = 1000;
lambda_init = [0;0];
alpha3 = 0.1;
[x_opt3, lambda_opt3, ~, ~] = dual_gradient_projection(lambda_init, alpha3, max_iter);

fprintf('Optimal solution: [%f, %f]\n', x_opt3(1), x_opt3(2));
fprintf('Optimal dual variable: [%f, %f]\n', lambda_opt3(1), lambda_opt3(2));
fprintf('Objective value: %f\n', f0(x_opt3(1), x_opt3(2)));

% primal feasibility
constraints = [-2*x_opt3(1)-x_opt3(2)+3; -x_opt3(1)-2*x_opt3(2)+3];
primal_feasible = all(constraints <= tol);
fprintf('Primal feasibility: %s\n', mat2str(primal_feasible));

% dual feasibility
dual_feasible = all(lambda_opt3 >= 0);
fprintf('Dual feasibility: %s\n', mat2str(dual_feasible));

% Complementary slackness
complementary_slackness = all(abs(lambda_opt3 .* constraints) <= tol);
fprintf('Complementary slackness: %s\n', mat2str(complementary_slackness));

gradient = [2*x_opt3(1)+3*x_opt3(2)+2-2*lambda_opt3(1)-lambda_opt3(2);
            3*x_opt3(1)+18*x_opt3(2)-5-lambda_opt3(1)-2*lambda_opt3(2)];
fprintf('Gradient of Lagrangian function: [%f, %f]\n', gradient(1), gradient(2));
gradient_zero = all(abs(gradient) <= tol);
fprintf('Gradient of Lagrangian function is zero: %s\n', mat2str(gradient_zero));
optimal = primal_feasible && dual_feasible && complementary_slackness && gradient_zero;
fprintf('Solution is optimal: %s\n', mat2str(optimal));


if optimal
    fprintf('Solution is optimal: Problem 1 ends. Thanks!!')
end 



