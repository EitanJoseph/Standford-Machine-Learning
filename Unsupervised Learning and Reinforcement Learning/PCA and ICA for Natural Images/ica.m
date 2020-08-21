function W = ica(X)

[n, m] = size(X);
% The chunk size will indicate how many training examples we will compute
% at each iteration of gradient descent.
% We will use 100 examples at a time
chunk = 100;
% Initialize W as the identity matrix in nxn
W = eye(n);
% our learning rate, alpha, is chosen to be 0.0005
alpha = 0.0005;
% Loop over 10 iterations
for c = 1:10
    %randomize the order of the examples
    X = X(:,randperm(m));
    % iterate over each chunk of the training set
    for i = 1:floor(m/chunk)
        % partition X into chunk sized subsets to be used with the gradient
        % descent. This way we only need to compute ~chunk~ values at any
        % sub-iteration
        x = X(:,(i-1)*chunk+1:i*chunk);
        % We can compute each sigmoid element-wise 
        sigmoidVector =  (1 - 2./(1+exp(-W*x)));
        % Finally we use the stochastic gradient descent formula to compute
        % the next iteration's W matrix
        W = W + alpha * (sigmoidVector * x' + chunk * inv(transpose(W)));
    end
end
        

