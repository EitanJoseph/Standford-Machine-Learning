function U = pca(X)
% Dimensions of X
[m,~]=size(X);

% Compute the covariance matrix sigma
sigma = (1/m) .* (X * transpose(X));

% Extract the eigenvalues of the covariance matrix sigma and place them on
% the diagonal of D
% Extract the associated eignvectors and place them as columns of matrix Us
[Us, D] = eig(sigma);

% The result of eig(sigma) returns a matrix of eigenvectors that is not
% sorted properly - we therefore need to ensure that it is in descending order

% First we grab the original indices of the eigenvalues and sort them in
% descending order
[~, ind] = sort(diag(D), "descend");
% Next we sort U's column vectors with respect to the indices of D
U = Us(:, ind);
