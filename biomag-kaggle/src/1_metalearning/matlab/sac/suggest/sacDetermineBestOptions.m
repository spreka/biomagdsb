function [SEARCH bayesOptimal] = sacDetermineBestOptions(SEARCH, c, M, baseline, bayesOptimal) 

global logFID

%pmetrics = metrics;
%pweights = ones(1,numel(metrics))/numel(metrics);

oldOptimal = bayesOptimal;

%[P bayesOptimal] = sacComputePerformanceIndex(SEARCH(c).E, pmetrics, pweights, SEARCH(c).status, baseline, bayesOptimal, 1);
[P E bayesOptimal] = sacComputePerformanceIndex(SEARCH(c).E, M, SEARCH(c).status, baseline, bayesOptimal,1);
SEARCH(c).P = P;
SEARCH(c).E = E;

% if a new bayesOptimal was found
if ~isequal(bayesOptimal,oldOptimal)
    inds = 1:numel(SEARCH);
    inds = setdiff(inds, c);
    for i = inds
        [P E] = sacComputePerformanceIndex(SEARCH(i).E,M,SEARCH(i).status,baseline,bayesOptimal);
        SEARCH(i).P = P;
        SEARCH(i).E = E;
    end
end




% find the best performer according to P
mxP = max(SEARCH(c).P);

SEARCH(c).bestInds = find(SEARCH(c).P == mxP);
numBest = numel(SEARCH(c).bestInds);



% report the best performance index
sacLog(3,logFID,' %d best configurations found for %s. First is given below:\n', numBest, SEARCH(c).classifier);
sacLog(3,logFID,'   %s %s\n', SEARCH(c).classifier, SEARCH(c).cmds{SEARCH(c).bestInds(1)});


% % only print the first
% for j = 1:length(SEARCH(c).bestInds)
%     sacPrintMethodPerfomance( SEARCH(c).classifier, logFID, M.criteria, SEARCH(c).E(SEARCH(c).bestInds(j)).performanceValues, SEARCH(c).P(SEARCH(c).bestInds(j)) );
% end

% sacLog(3,logFID,' %d best option configurations found:\n', numBest);
% for j = 1:length(SEARCH(c).bestInds)
%     sacPrintMethodPerfomance( SEARCH(c).classifier, logFID, M.criteria, SEARCH(c).E(SEARCH(c).bestInds(j)).performanceValues, SEARCH(c).P(SEARCH(c).bestInds(j)) );
% end


% for j = 1:length(SEARCH(c).bestInds)
%     sacLog(3,logFID,'  %d/%d option string: %s\n    ', j, numBest, SEARCH(c).cmds{SEARCH(c).bestInds(j)});
%     sacLog(3,logFID,'P:%1.4f   ', SEARCH(c).P(SEARCH(c).bestInds(j)));
%     for m = 1:numel(metrics)
%         sacLog(3,logFID,'%s:%1.4f ', metrics{m}, SEARCH(c).metrics(SEARCH(c).bestInds(j),m));
%     end
%     sacLog(3, logFID, '\n');
% end



