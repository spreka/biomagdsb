function SEARCH = sacSuggest(d, varargin)
% SEARCH = sacSuggest(D,...)
% 
% Performs a search through a list of classification methods and their
% associated paramters for the best performing classification method and
% configuration. The only required input is D, a dataset to optimize for,
% which can be loaded using sacReadArff. Other possible options to
% configure the search include:
%  
%  To set the time spent optimizing a given classifier:
%    sacSuggest(D,'timeLimit',120,...)  
%
%  To select the method used for optimization:
%    sacSuggest(D,'random',...)     
%    sacSuggest(D,'sa',...)
%    sacSuggest(D,'default',...)
%
%  To specify the criteria to optimize over (optionally including weights
%  for specific measures):
%    sacSuggest(D,'criteria',{'ACC','BEP'},...)       
%    sacSuggest(D,'criteria','AUC',...)               
%    sacSuggest(D,'criteria',{'ACC','CONFUSION'},'weights',{1, [.25 -.25;-.25 .25]},...) 
%
%  To search using the distributed computing mode (recommended):
%    sacSuggest(D,'parallel',...) 
%    sacSuggest(D,'distributed',...) 
%
%  To search in serial (non-distributed) mode (not recommended):
%    sacSuggest(D,'serial',,...)
%
%  To specify the classifier or list of classifiers you wish to search:
%    sacSuggest(D,{'KNN_Weka','SVM_Libsvm'},...)
%    sacSuggest(D,'KNN_Weka',...)
%
%  To specify which measures to compute in addition to the optimization
%  criteria (such as an ROC curve, PR curve, confusion matrix):
%    sacSuggest(D,'measures', {'ACC','ROC'},'weights',{1,2}...) 
%    sacSuggest(D,'measures', 'PR',...)
%
%  To specify what types of messages get displayed to the consol/logfile
%    sacSuggest(D,'displaymessagepriority', 3, ...); % [3 is default]
%    sacSuggest(D,'logMessagePriority',5,...); % [5 is default]
%
%  sacSuggest(d,{'SVM_Weka','SVM_Libsvm'}, 'measures', {'ACC', 'AUC', 'PR'}, 'criteria', {'CONFUSION','ACC','BEP','AUC'}, 'weights',{1,[],2,5});
%  sacSuggest(d,{'SVM_Weka','SVM_Libsvm'}, 'measures', {'ACC', 'AUC', 'PR', 'TPR'}, 'criteria', {'ACC','AUC','BEP'}, 'weights',{1,2,[]});
% See also: sacPredict, sacTrain, sacReadArff


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


% TODO: pass options to sub-suggestion functions?
% TODO: should be pass back the best trained model(s)?
% TODO: make sure the cmd strings are unique - or we might waste a lot of time!
% TODO: support stacking
% TODO: generate a report


global SACROOT
global logFID
global logDIR
global displayMessagePriority
global logMessagePriority %#ok<*NUSED>
global baseline
global bayesOptimal
global absoluteOptimal


% create a folder and and logfile ID (global variables logDIR, logFID)
createLogfileFolder()

% parse the inputs
[timeLimit,searchMethod,pll,classifiers,M,localfilenm] = sacSuggestParseInputs(varargin);

% initialize the search structure where we will store our results
SEARCH = initSearch();

% compute the baseline and Bayes optimal (initial estimate) performance levels
sacLog(3,logFID,' Estimating baseline performance level...\n');
sacGetBaselineAndOptimal(d,classifiers,M,timeLimit,pll);
sacLog(3,logFID,' Done.\n');



for c = 1:numel(classifiers)
    clear E cmds OPTs status;
    sacLog(3,logFID,'\n=========================== searching %s (%d/%d) ===========================\n', classifiers{c}, c, numel(classifiers));
    
    switch lower(searchMethod)
        % --------------------------- Random Search --------------------------------
        case 'random'
            sacLog(3,logFID,' Random search (%ds time limit)...\n', timeLimit);
            if pll
                [E cmds OPTs status] = sacSuggestRandomDistributed(d, classifiers{c}, timeLimit, M); 
            else
                [E cmds OPTs status] = sacSuggestRandomSerial(d, classifiers{c}, timeLimit, M);
            end
        % --------------------------- Simulated Annealing Search --------------------------------    
        case {'simulatedannealing', 'sa'}
            sacLog(3,logFID,' Simulated annealing search (%ds time limit)...\n', timeLimit);
            if pll
                [E cmds OPTs status] = sacSuggestSimulatedAnnealDistributed(d, classifiers{c}, timeLimit, M); 
            else
                error('Serial mode not yet supported (sorry).');
            end
        % --------------------------- Default --------------------------------    
        case {'default'}
            sacLog(3,logFID,' Default classification parameters (%ds time limit)...\n', timeLimit);
            if pll
                
            else
                error('Serial mode not yet supported (sorry).');
            end
        % --------------------------- Otherwise --------------------------------      
        otherwise
            error('Unknown search method');            
    end
    
    
    SEARCH(c).classifier = classifiers{c};
    SEARCH(c).cmds = cmds; SEARCH(c).OPTs = OPTs; SEARCH(c).E = E; SEARCH(c).status = status; %#ok<*AGROW>
    
    sacLog(3,logFID,' %d/%d total configurations were successfully evaluated\n', sum(SEARCH(c).status), numel(SEARCH(c).status));
    sacLog(3,logFID,'  - - - - - - - - - - - - - results for %s - - - - - - - - - - - - -\n', classifiers{c});
    
    [SEARCH bayesOptimal] = sacDetermineBestOptions(SEARCH, c, M, baseline, bayesOptimal);
    
    
    
    sacLog(3,logFID, '\n');
    sacPrintMethodPerfomance('baseline',logFID,M.criteria,baseline.performanceValues,0)
    sacPrintMethodPerfomance('bayesOptimal',logFID,M.criteria,bayesOptimal.performanceValues,1)
    %sacPrintMethodPerfomance('absoluteOptimal',logFID,M.criteria,absoluteOptimal.performanceValues,1)
    
    
    % display the best performance for each classifier evaluated thus far
    sacLog(3,logFID, '    ........   ................................................................................\n');
    for i = 1:numel(SEARCH)
        bestInd = SEARCH(i).bestInds(1);
        % raw metrics
        sacPrintMethodPerfomance(SEARCH(i).classifier,logFID,M.criteria, SEARCH(i).E(bestInd).performanceValues, SEARCH(i).P(bestInd));
        % calibrated metrics
        %sacPrintMethodPerfomance(SEARCH(i).classifier,logFID,M.criteria, SEARCH(i).E(bestInd).performanceCalibrated, SEARCH(i).P(bestInd) );
    end
    
    
    
    search = SEARCH(c);
    search = rmfield(search, 'OPTs');
    numbest = min(10, numel(search.bestInds));
    bestInds = search.bestInds(1:numbest)';
    search.E = search.E([1 bestInds]);
    search = rmfield(search, 'metrics');
    S(c) = search; %#ok<NASGU>
    
    % save output files
    savefilenm = [logDIR 'SACsearch.mat'];
    %save(savefilenm, 'SEARCH', 'd', 'baseline', 'bayesOptimal');
    save(savefilenm, 'S', 'd', 'baseline', 'bayesOptimal');
    sacLog(3,logFID, '\n Saved results to %s\n', savefilenm);
    save(localfilenm, 'S', 'd', 'baseline', 'bayesOptimal');
    sacLog(3,logFID, '\n Saved results to %s\n', localfilenm);
end


sacLog(3,logFID,'\n\n- - - - - - - - - - - - - - - - - - - - - - - - - -\n');
sacLog(3,logFID,'               OPTIMIZATION COMPLETE!\n');
sacLog(3,logFID,'- - - - - - - - - - - - - - - - - - - - - - - - - -\n');
fclose('all');

















% -------------------------- SUPPORTING FUNCTIONS -----------------------

function createLogfileFolder()

global SACROOT
global logFID
global logDIR

% get a FID for the logfile, and create a folder to put it in
logDIR = datestr(now, 'yyyy-mmm-dd-HH.MM.SS./');
logDIR(15) = 'h';  logDIR(18) = 'm'; logDIR(21) = 's';
logDIR = [SACROOT '/logs/' logDIR];
mkdir(logDIR);
logFID = fopen([logDIR 'sacSuggest.log'], 'w');







function SEARCH = initSearch()
SEARCH.E = []; 
SEARCH.cmds = []; 
SEARCH.OPTs = []; 
SEARCH.status = []; 
SEARCH.metrics = []; 
SEARCH.P = [];
SEARCH.bestInds = [];
SEARCH = orderfields(SEARCH);

