function [E cmds OPTs status] = sacSuggestRandomDistributed(d, classifier, timeLimit, M)

global logFID  % the file id of the logfile



% package a few variables into Z to keep function calls clean
Z.sched = findResource('scheduler', 'type', 'local');       % job manager
                                        % TODO: allow non-local job managers
Z.clusterSize = Z.sched.ClusterSize;    % number of available workers  
Z.taskOverheadRatio = 20;               % amount of "task time" for a job (compared to overhead)
Z.waitFactor = 4; %3                    % how long we are willing to wait for a job to finish before we cancel it
                                        % TODO: get this value from a config file!
Z.paths = sacGetPaths;                  % path dependencies for workers
Z.classifier = classifier;              % the classifier to test
Z.M = M;                                % the metrics to test

tRemain = timeLimit;                    % remaining time until time limit is reached
timeUpFlag = 0;                         % indicates if the time limit has been reached
taskTime = 0;                           % running estimate of the runtime of a typical task
jobOverhead = 0;                        % estimate of the overhead associated with sending a distributed job
checkInterval = .1;                     % how often to check status of the jobs
searchMode = 'random';                  % the type of search to use

% submit an initial set of jobs (with 1 task each and no time limit) to
% fill the clusterSize. The 1st task contains default string
J = sacCreateDistributedJob(d,Z,'default',1,tRemain);
for j = 2:Z.clusterSize
    J(j) = sacCreateDistributedJob(d,Z,searchMode,1,tRemain);
end


% repeatedly check the status of the jobs until the time limit has been
% reached
t0 = tic; 
while 1
    % determine how much time we have left to search
    tElapsed = toc(t0);  tRemain = timeLimit - tElapsed;
    
    % check the status of the jobs. If there is a free worker, add a job.
    %[J taskTime jobOverhead] = sacCheckJobStatus(J, d, taskTime, tRemain, Z, logFID, metrics);    
    [J taskTime jobOverhead] = sacCheckJobStatus(J,Z,d,searchMode,tRemain,1,taskTime,jobOverhead);
    
    if tElapsed > timeLimit
        numRunningJobs = sum([J(:).running]);
        if ~timeUpFlag
            for j = find([J(:).running]) 
                runningTime = getRunTime(J(j).job);
                sacLog(4,logFID,'     Time is up! cancelling job %d, time limit exceeded\n', J(j).job.ID);
             	fid = fopen([J(j).logfile], 'a');
              	sacLog(4,fid,'    cancelled job %d, runningTime (%0.5gs) > timeLim (%0.5gs)\n', J(j).job.ID, runningTime, J(j).timeLim);
                cancelJob( J(j).job );
            	J(j).running = 0;
              	J(j).finished = 0;
              	J(j).cancelled = 1;
            end
            %sacLog(4,logFID,'   Time is up! waiting for %d remaining jobs to finish\n', numRunningJobs);
            timeUpFlag = 1;
        end
        if numRunningJobs == 0
            break;
        end
    end
    pause(checkInterval);  % check job status periodically
end


tElapsed = toc(t0);
sacLog(3,logFID,' Finished in %0.5gs\n', tElapsed);


% % check to see if any Jobs failed
% sacLog(4,logFID,'   Checking for errors in the Jobs\n');
% for j = 1:numel(J)
%     if ~isempty(J(j).job.Tasks.ErrorIdentifier)
%         if strcmpi(J(j).job.Tasks.ErrorIdentifier,'distcomp:task:Cancelled')
%             %fprintf('   Job %d cancelled\n', J(j).job.ID);
%         else
%             %J(j).job.tasks
%             sacLog(2,logFID,'   Warning: An error occured in Job %d!\n', J(j).job.ID);
%         end
%     end
% end


% retreive the results
sacLog(4, logFID,'   Retreiving results from distributed jobs\n');
[cmds OPTs E status] = sacGetDistributedJobResults(J,d);


% clean up: destroy the jobs
sacDestroyJobs(Z.sched);



function runTime = getRunTime(job)

% sometimes the job startime is not available
tStart = [];
while isempty(tStart)
    tStart = job.StartTime;
    pause(0.05);
end
tStart = strrep(tStart, 'CET', '');
tStart = regexprep(tStart, '[ \t\r\f\v]+', ' ');
tStart = tStart(5:end);
dv = datenum(tStart, 'mmm dd HH:MM:SS yyyy');

tRun = now - dv;
tRun = datevec(tRun);

runTime = tRun(6) + 60*tRun(5) + 60*60*tRun(4) + 24*60*60*tRun(3);




function cancelJob(job)

cancel(job);
pause(0.1);  % added a pause to avoid spamming the job manager