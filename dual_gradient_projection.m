function [x_opt, lambda_opt, x_history, lambda_history] = dual_gradient_projection(lambda_init, alpha, max_iter)
    tol = 1e-15;

    % Initialize history
    x_history = zeros(max_iter, 2);
    lambda_history = zeros(max_iter, 2);

    % Initialize the dual variables
    lambda = lambda_init;

    for k = 1:max_iter
        % Compute x1 and x2 based on the current lambda values
        x1 = (-17 + 11*lambda(1) + 4*lambda(2)) / 9;
        x2 = (16 - 4*lambda(1) + lambda(2)) / 27;

        x1 = max(0, x1);
        x2 = max(0, x2);

        % Calculate the gradient of the dual function
        grad_g = [-2*x1 - x2 + 3; -x1 - 2*x2 + 3];

        % Update the dual variables
        lambda_new = lambda + alpha * grad_g;
        
        % Project the updated dual variables onto the non-negative orthant
        lambda_new = max(lambda_new, 0);
        
        % Store history
        x_history(k, :) = [x1; x2];
        lambda_history(k, :) = lambda;

        % Check convergence
        if norm(lambda_new - lambda) < tol
            x_history = x_history(1:k, :);
            lambda_history = lambda_history(1:k, :);
            break;
        end

        % Update lambda for the next iteration
        lambda = lambda_new;
    end

    % Compute the optimal primal solution
    x_opt = [(-17 + 11*lambda(1) + 4*lambda(2)) / 9; (16 - 4*lambda(1) + lambda(2)) / 27];

    % Ensure the non-negativity constraints are satisfied
    x_opt = max(x_opt, 0);

    % Return the optimal dual solution
    lambda_opt = lambda;
end