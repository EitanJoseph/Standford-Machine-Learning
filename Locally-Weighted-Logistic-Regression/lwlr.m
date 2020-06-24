function y = lwlr(X_train, y_train, x, tau)
[m,n]=size(X_train);
Theta = zeros(n,1);

% Building W
W = exp(-sum((X_train - repmat(x', m, 1)).^2, 2) / (2*tau^2));
lambda = 0.0001;

% Building grad_theta
grad_theta = ones(n,1);
while(grad_theta > 1e-6*ones(n,1))
    %Building h
    h = 1./(1.+exp(-X_train*Theta));
    %Building Z
    Z = W.*(y_train - h);
    %Buildig grad_theta
    grad_theta = transpose(X_train)*Z - lambda*Theta;
    %Building Hessian
    d = -W.*h.*(1-h);
    D = diag(d);
    Hessian = transpose(X_train)*D*X_train - lambda*eye(n,n);
    %Updating Theta
    Theta = Theta - inv(Hessian)*grad_theta;
end

newH = 1./(1.+exp(-transpose(Theta)*x));
y = newH > 0.5;

end
