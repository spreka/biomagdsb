function [baseline bayesOptimal] = sacGetBaselineAndOptimal(d,classifiers,M,timeLimit,pll)


% From the Suggest a Classifier Library (SAC), a Matlab Toolbox for cell
% classification in high content screening. http://www.cellclassifier.org/
% Copyright © 2011 Kevin Smith and Peter Horvath, Light Microscopy Centre 
% (LMC), Swiss Federal Institute of Technology Zurich (ETHZ), Switzerland. 
% All rights reserved.
%
% This program is free software; you can redistribute it and/or modify it 
% under the terms of the GNU General Public License version 2 (or higher) 
% as published by the Free Software Foundation. This program is 
% distributed WITHOUT ANY WARRANTY; without even the implied warranty of 
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
% General Public License for more details.

global logFID  
global SACROOT
global baseline
global bayesOptimal


% define the baseline method(s)
baselineClassifier = 'DecisionTable_Weka';   %'NaiveBayes_Weka';


switch pll
    % SERIAL PROCESSING MODE
    case 0
        % get the baseline performance
        [J.cmds J.OPTs] = sacGetDefaultOptions(baselineClassifier);
        J.logfile = logFID;
        baseline = sacSuggestList(J.cmds,d,baselineClassifier,M,SACROOT,J.logfile);
        
        % get the optimal performance
        Eblank = sacEval([],[],d,'Empty');
        [E(1:numel(classifiers))] = deal(Eblank);
        for c = 1:numel(classifiers)
            [J(c).cmds J(c).OPTs] = sacGetDefaultOptions(classifiers{c}); %#ok<AGROW>
            J(c).logfile = logFID; %#ok<AGROW>
            E(c) = sacSuggestList(J(c).cmds,d,classifiers{c},M,SACROOT,J(c).logfile);
        end
        
        bayesOptimal = sacEval([],[],d,'Empty');
        bayesOptimal.performanceCriteria = E(1).performanceCriteria;
        bayesOptimal.performanceCoeffs = E(1).performanceCoeffs;

        for m = 1:numel(M.criteria)
            criteriaField = M.criteria{m};
            weightField = ['W' lower(criteriaField)];
            weightedPerformance = zeros(1,numel(E));
            % the sign is included to flip signs of metrics such as FPR,RMS
            % to ensure that high values indicate better performance
            for i = 1:numel(E)
                weightedPerformance(i) = E(i).performanceValues(m)*sign(E(i).performanceCoeffs(m));
            end
            [bestVal,bestInd] = max(weightedPerformance); clear bestVal; %#ok<ASGLU>
            bayesOptimal.performanceValues(m) = E(bestInd).performanceValues(m);
            bayesOptimal.(criteriaField) = E(bestInd).(criteriaField);
            bayesOptimal.(weightField) = E(bestInd).(weightField);
        end
        
        
        
	% DISTRIBUTED PROCESSING MODE
    case 1
        % ---- baseline performance ----
        
        % TODO: allow non-local job managers
        % package a few variables into Z to keep function calls clean
        Z.sched = findResource('scheduler', 'type', 'local'); % job manager
        Z.clusterSize = Z.sched.ClusterSize;% number of available workers  
        Z.taskOverheadRatio = 20;           % amount of "task time" for a job (compared to overhead)
        Z.waitFactor = 3;                   % how long we are willing to wait for a job to finish before we cancel it
                                            % TODO: get this value from a config file!
        Z.paths = sacGetPaths;              % path dependencies for workers
        Z.classifier = baselineClassifier;  % the classifier to test
        Z.M = M;                            % the metrics to test

        % create the job to compute baseline performance
        J = sacCreateDistributedJob(d,Z,'default',1,timeLimit);
        

        
        
        
        % ---- bayesOptimal performance ----
        
%       	for c = 1:numel(classifiers)
%             Z.classifier = classifiers{c};
%             J(c+1) = sacCreateDistributedJob(d,Z,'default',1,timeLimit);
%         end
        
        % repeatedly check the status of the job
        checkInterval = .1;  t0 = tic;
        while 1
            % determine how much time we have left to search
            tElapsed = toc(t0);
            tRemain = timeLimit - tElapsed;
            % check the status of the job
            J = sacCheckJobStatus(J,Z,d,'default',tRemain,0);
            % break the loop when the job is finished
            numRunningJobs = sum([J(:).running]);
            if numRunningJobs == 0
                break;
            end
            pause(checkInterval);  % check job status periodically
        end
        
        
        
%      	% repeatedly check the status of the various jobs
%         checkInterval = .1;  t0 = tic;
%         while 1
%             % determine how much time we have left to search
%             tElapsed = toc(t0);
%             tRemain = timeLimit - tElapsed;
%             % check the status of the jobs
%             J = sacCheckJobStatus(J,Z,d,'default',tRemain,0);
%             % break the loop when the job is finished
%             numRunningJobs = sum([J(:).running]);
%             if numRunningJobs == 0
%                 break;
%             end
%             pause(checkInterval);  % check job status periodically
%         end
        
        % retrieve the results
        [cmds OPTs E status] = sacGetDistributedJobResults(J,d); %#ok<ASGLU>
        %clear cmds OPTs;
        
        
        % determine the baseline performance
        if status(1) == 0
            error('Baseline method did not successfully evaluate');
        else
            baseline = E(1);
        end
        
        
        % determine the bayesOptimal performance
        
        bayesOptimal = baseline;
        for i = 1:numel(bayesOptimal.performanceCriteria)
            if ismember(bayesOptimal.performanceCriteria{i}, {'ACC', 'AUC', 'BEP', 'FSC', 'LFT', 'PRE', 'REC', 'TPR'});
                bayesOptimal.performanceValues(i) = bayesOptimal.performanceValues(i) * 1.01;
            elseif ismember(bayesOptimal.performanceCriteria{i}, {'ERR', 'FPR', 'MXE', 'RMS'});
                bayesOptimal.performanceValues(i) = bayesOptimal.performanceValues(i) * .99;
            end
        end
        
        % SET THE RMS baseline to very bad probability predictions
        for i = 1:numel(baseline.performanceCriteria)
            if ismember(baseline.performanceCriteria{i}, {'RMS'});
                baseline.performanceValues(i) = .25; 
            end
        end
        
        
%         % determine the bayesOptimal performance
%         if sum(status(2:end)) == 0
%             error('BayesOptimal could not be computed because classifiers failed to finish in time');
%         end
%         bayesOptimal = sacEval([],[],d,'Empty');
%         bayesOptimal.performanceCriteria = E(1).performanceCriteria;
%         bayesOptimal.performanceCoeffs = E(1).performanceCoeffs;
%         
%         clear weightedPerformance;
%         for m = 1:numel(M.criteria)
%             criteriaField = M.criteria{m};
%             weightField = ['W' lower(criteriaField)];
%             weightedPerformance = zeros(1,numel(E));
%             % the sign is included to flip signs of metrics such as FPR,RMS
%             % to ensure that high values indicate better performance
%             for i = 1:numel(E)
%                 if status(i) == 1
%                     weightedPerformance(i) = E(i).performanceValues(m)*sign(E(i).performanceCoeffs(m));
%                 end
%             end
%             [bestVal,bestInd] = max(weightedPerformance); clear bestVal; %#ok<ASGLU>
%             bayesOptimal.performanceValues(m) = E(bestInd).performanceValues(m);
%             bayesOptimal.(criteriaField) = E(bestInd).(criteriaField);
%             bayesOptimal.(weightField) = E(bestInd).(weightField);
%         end
end


% TODO: add a check to see if the baseline is equal to bayesOptimal for any
% performance criteria - this could cause problems with later evaluations.
% Also, possibly hand-set the baseline for certain metrics


% display the baseline and optimal performance
%sacPrintMethodPerfomance('baseline',logFID,M.criteria,baseline.performanceValues,0)
%sacPrintMethodPerfomance('bayesOptimal',logFID,M.criteria,bayesOptimal.performanceValues,1)


