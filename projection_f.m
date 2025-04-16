function x = projection_f(y)
    A = [-2 -1; -1 -2; -1 0; 0 -1];
    b = [-3; -3; 0; 0];
    % Solver for quadratic objective functions with linear constraints
    n = length(y);
    H = eye(n);
    x0 = y;
    options = optimset('Display','off');
    x = quadprog(H,-y,A,b,[],[],[],[],x0,options); %quadprog finds a minimum for a problem specified by
end

