function [intervals] = splitToIntervals(n,weights)
    M = length(weights);
    intervals = ones(1,M+1);
    proportionalWeights = weights./sum(weights);    
    for i=2:M
        intervals(i) = intervals(i-1)+round(proportionalWeights(i-1)*n);
    end
    intervals(M+1) = n+1;
end