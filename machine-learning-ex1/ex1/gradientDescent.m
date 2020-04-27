function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,
  newtheta = zeros(size(theta),1);
  for index = 1:size(theta),
    sum = 0;
    
    for sample = 1:m,
      sum += (X(sample,:)*theta -y(sample))*X(sample,index);
    endfor
    
    newtheta(index,1) = theta(index,1) - alpha/m*sum;
    
  endfor
  theta = newtheta;
  J_history(iter,1) = computeCost(X, y, theta);
endfor

%plot(J_history);
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    






    % ============================================================

    % Save the cost J in every iteration    
    %J_history(iter) = computeCost(X, y, theta);
end
