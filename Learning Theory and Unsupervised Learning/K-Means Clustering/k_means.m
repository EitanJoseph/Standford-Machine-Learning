function [clusters, centroids] = k_means(X, k)

[m,n]=size(X);

% Initial centroids are random chosen depending on the range of X
% In this case the range of X is [-1,1]
centroids = 2*rand(k,n) - 1;
clusters = zeros(m,1);
disp("processing");
flag = 1;
oldCentroids = centroids;
while 1==flag
    oldOldcentroids = oldCentroids;
    oldCentroids = centroids;
    % Step 1: classify each Xi to some cluster
    for i = 1:m
        % Find the closest cluster
        min_dist = intmax;
        min_centroid = -1;
        for j = 1:k
            if min_dist > norm(X(i)-centroids(j))
                min_centroid = j;
                min_dist = norm(X(i)-centroids(j));
            end
        end
        % And place X(i) into that cluster
        clusters(i) = min_centroid;
    end
    % Step 2: recenter the each cluster centroid
    for j = 1:k
        % Compute the total "location" or "value" of each member of cluster j
        sum = zeros(1,n);
        for i = 1:m
            if clusters(i) == j
                sum = sum + X(i,:);
            end
        end
        % Average out their overall "location" or "value"
        reg = 0;
        for i = 1:m
            if clusters(i) == j
                reg = reg + 1;
            end
        end
        if reg ~= 0
            sum = sum./reg;
        end
        % set centroid j to its new "location" or "value"
        centroids(j,:) = sum;
    end
    
    % We have a two part condition here due to the fact that k-means
    % clustering can often oscillate between two centers without ever
    % converging. Therefore we take the extra step of checking the distance
    % from both the previous and previous-previous iterations
    if (abs(oldCentroids-centroids) <= 0.0001*ones(k,n) | abs(oldOldcentroids - centroids)<0.0001*ones(k,n));
        flag = 0;
    end
end
disp("finished");