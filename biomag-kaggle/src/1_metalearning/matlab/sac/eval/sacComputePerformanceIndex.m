function [P E bayesOptimal] = sacComputePerformanceIndex(E, M, status, baseline, bayesOptimal, updateOptimalFlag)
%
% [P E bayesOptimal] = sacComputePerformanceIndex(E, M, status, baseline, bayesOptimal, updateOptimalFlag)
%
%
%


% extract the performanceCriteria values from evaluated classifiers, E
performanceValues = NaN(numel(E), numel(M.criteria));



for i = 1:numel(E)
    if status(i) == 1
        performanceValues(i,:) = E(i).performanceValues .* sign(E(i).performanceCoeffs);    
    end
end


% update the bayesOptimal performance, if necessary (and requested)
newBayesOptimal = 0;
if exist('updateOptimalFlag', 'var')
    if updateOptimalFlag
        [bestVal bestInd] = max(performanceValues, [], 1);
        for m = 1:numel(M.criteria)
            if bestVal(m) > sign(bayesOptimal.performanceCoeffs(m)) * bayesOptimal.performanceValues(m)
                bayesOptimal.performanceValues(m) = E(bestInd(m)).performanceValues(m);
                criteriaField = M.criteria{m};
                weightField = ['W' lower(criteriaField)];
                bayesOptimal.(criteriaField) = E(bestInd(m)).(criteriaField);
                bayesOptimal.(weightField) = E(bestInd(m)).(weightField);
                
                newBayesOptimal = 1;
                
            end
        end
    end
end
%if newBayesOptimal
%    sacPrintMethodPerfomance('bayesOptimal',logFID,M.criteria,bayesOptimal.performanceValues,1)
%end            



% compute the performance index (take into account weights between
% different criteria, performanceCoeffs)
P = -Inf * ones(numel(E),1);
for i = 1:numel(E)
    Pi = -Inf * ones(1,numel(M.criteria));
    if status(i) == 1
        vals = E(i).performanceValues;
        base = baseline.performanceValues;
        opti = bayesOptimal.performanceValues;
        sgns = sign(E(i).performanceCoeffs);
        coef = abs(E(i).performanceCoeffs);
        
        Pi = ((sgns.*vals) - (sgns.*base))./((sgns.*opti) - (sgns.*base));
        E(i).performanceCalibrated = Pi;
        Pi = coef .* Pi;
    end
    P(i) = sum(Pi);
end