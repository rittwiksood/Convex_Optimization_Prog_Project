% Gradient descent algorithm
function [x1_min, x2_min, f_min, x1_history, x2_history] = gradient_descent(f, grad_f, gamma, x1_0, x2_0, max_iter)
    % Initialize variables
    x1 = x1_0;
    x2 = x2_0;
    x1_history = [x1];
    x2_history = [x2];
    x1_min = x1;
    x2_min = x2;
    f_min = f(x1, x2);
    
    % Iteratively update x1 and x2
    for k = 1:max_iter-1
        grad = grad_f(x1, x2);
        x1_new = x1 - gamma*grad(1);
        x2_new = x2 - gamma*grad(2);
        x1 = x1_new;
        x2 = x2_new;
        x1_history = [x1_history, x1];
        x2_history = [x2_history, x2];
        f_val = f(x1, x2);
        if f_val < f_min
            f_min = f_val;
            x1_min = x1;
            x2_min = x2;
        end
    end
end
