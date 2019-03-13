function J = sacCreateDistributedJob(d,Z,searchMode,numTasks,varargin)
%
% d = dataset
% Z.sched = job scheduler
% Z.clusterSize = number of available workers
% Z.waitfactor = amount of "task time" for a job compared to overhead
% Z.paths = path dependencies for the workers (sacGetPaths)
% Z.classifer = the classifier to test
% Z.M = the metrics to test (M.measures, M.criteria)
% searchMode = the type of search (random, simulatedannealing, etc)
% numTasks = the number of tasks for the job can be numeric (>= 1) or set to be automatic 'auto'
% (optional) timeLim
% (optional) tRemain
% (optional) taskTime
% (optional) jobOverhead
%
% opional ways of calling:
% sacCreateDistributedJob(d,Z,searchMode,numTasks,timeLim)
% sacCreateDistributedJob(d,Z,searchMode,numTasks,tRemain,taskTime,jobOverhead)


global SACROOT
global logFID
global logDIR



% handle the two ways of calling sacCreateDistributedJob
if nargin == 5
   timeLim = varargin{1}; 
   tRemain = timeLim;
   taskTime = tRemain;
   jobOverhead = 0;
else
    tRemain = varargin{1};
    taskTime = varargin{2};
	jobOverhead = varargin{3};
end


% if the number of tasks is not specified in the function call, determine it dynamically
switch numTasks
    case 'auto'
        if ~exist('taskTime', 'var')
            error('Automatic selection of the number of tasks requires the taskTime to be provided');
        else
            taskTime = min(taskTime, abs(tRemain));
            desiredNumTasks = max(1, round( (Z.taskOverheadRatio*jobOverhead)/taskTime));  % round or floor?

            % estimated time to complete the desired number of tasks
            J.eta = desiredNumTasks*taskTime + jobOverhead;

            % check to see if we will finish within tRemain, set numTasks accordingly
            if J.eta < tRemain
                J.numTasks = desiredNumTasks;
            else
                J.numTasks = max(1, round((abs(tRemain) - jobOverhead)/taskTime)); % round or floor?
            end
        end
    otherwise
        if isnumeric(numTasks)
            J.numTasks = max(1,numTasks);
        else
            error('Invalid number of tasks per job specified');
        end
end
        


% if a timeLimit was not given, estimate the time for the J to complete
if ~exist('timeLim', 'var'); 
    J.eta = J.numTasks*taskTime + jobOverhead; 
    % if the J will take longer than the remaining time, don't do it
    if  (J.eta > tRemain)
        J = [];
        return;
    end
else
    J.eta = timeLim;
end


% create the job object J, set dependencies
J.job = createJob(Z.sched);
set(J.job, 'FileDependencies', {'sacSuggestList.m', 'sacLog.m'});
set(J.job, 'PathDependencies', Z.paths);


% set timeLimit for the J (exceeding will cause J to be cancelled)
if ~exist('timeLim', 'var'); 
    J.timeLim = J.eta * Z.waitFactor; 
else
    J.timeLim = timeLim;
end
J.timeLim = min(J.timeLim, max([0 tRemain]));

    
% generate a cell list of commands to send
switch searchMode
    case 'default'
        [cmd J.OPTs] = sacGetDefaultOptions(Z.classifier);
        J.cmds = cmd; 
    case 'random'
        [J.cmds J.OPTs] = sacGenerateRandomOptions(Z.classifier, J.numTasks, 'rand');
    case 'simulatedannealing'
        [J.cmds J.OPTs] = sacGenerateRandomOptions(Z.classifier, J.numTasks, 'rand');
end

J.classifier = Z.classifier;

% create a log file for this job
J.logfile = [logDIR 'Job' num2str(J.job.ID) '.log'];



%-------------------------------------------------------------
% CREATE THE JOB
% createTask(job, function_to_call, number_of_outputs, cell_of_inputs)
switch lower(searchMode)
    case {'default', 'random'}
        createTask(J.job, @sacSuggestList, 2, {J.cmds,d,Z.classifier,Z.M,SACROOT,J.logfile});
    case 'simulatedannealing'
        createTask(J.job, @sacSuggestList, 2, {J.cmds,d,Z.classifier,Z.M,SACROOT,J.logfile});
    otherwise
        error(['Unknown search mode ' searchMode]);
end
%-------------------------------------------------------------
      


% set some bookkeeping parameters for the job
J.running = 1;
J.finished = 0;
J.cancelled = 0;
J.duration = 0;
J.submitTime = now;

% write to the logfile
sacLog(4,logFID,'   submitting Job %d. tasks = %d, tLim = %0.5gs, eta = %0.5gs, tRemaining = %0.5gs\n', J.job.ID, J.numTasks, J.timeLim, J.eta, tRemain);
sacLog(5,logFID,'   job logfile: %s\n', J.logfile);

% submit the job to the scheduler
submit(J.job);

% sort the fields of the job
J = orderfields(J);



