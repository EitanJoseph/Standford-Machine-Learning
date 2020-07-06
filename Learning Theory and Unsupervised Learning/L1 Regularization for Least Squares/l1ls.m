function theta = l1ls(X,y,lambda)

[m,n] = size(X);
theta = ones(n,1);
exitVector = 10^(-5)*ones(n);
disp("processing...");
while(1==1)
    ogTheta = theta;
    for i = 1 : n
        thetaBar = theta;
        thetaBar(i)=0;
        si_pos = max(0, inv(transpose(X(:,i))*X(:,i))*(transpose(y-X*thetaBar)*X(:,i)-lambda));
        si_neg = min(0, inv(transpose(X(:,i))*X(:,i))*(transpose(y-X*thetaBar)*X(:,i)+lambda));
        j_pos = 0.5*norm(X*thetaBar + X(:,i)*si_pos - y)^2 + lambda*norm(thetaBar, 1) + lambda*si_pos;
        j_neg = 0.5*norm(X*thetaBar + X(:,i)*si_neg - y)^2 + lambda*norm(thetaBar, 1) - lambda*si_neg;
        if j_pos < j_neg
            theta(i) = si_pos;
        else
            theta(i) = si_neg;
        end
    end
    if norm(abs(ogTheta - theta), 1) < norm(exitVector,1)
        break;
    end
end
disp("finished");