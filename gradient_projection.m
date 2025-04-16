function [x, f_val, x1_history, x2_history] = gradient_projection(f, grad_f, alpha, x, max_iter)
    % Initialize variables
    x1_history = [x(1)];
    x2_history = [x(2)];
    
    % Iteratively update x1 and x2
    for k = 1:max_iter-1
        f_val = f(x(1), x(2));
        grad_f_val = grad_f(x(1), x(2));
        grad_f_vec = [grad_f_val(1); grad_f_val(2)];
    
        x = x - alpha*grad_f_vec; % Gradient descent
        
        % Project the point onto the feasible set. Writinf one more func
        % here
        x = projection_f(x);
        x1_history = [x1_history, x(1)];
        x2_history = [x2_history, x(2)];

    end
end