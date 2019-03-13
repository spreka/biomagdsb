function p = sacGetPaths()


p = {'.'};


if ispc
    str = [';' path ';']; %pathdef;
    
    %pat = ';([^;])*;';
    %n = regexpi(str, pat, 'match');
    pat = ';';
    n = regexpi(str, pat, 'split');
    
    for i = 1:numel(n)
        if ~isempty(strfind(n{i}, 'sac'))
            p{end+1} = n{i}; %#ok<*AGROW>
        end
    end
else
    str = [':' path ':']; %pathdef;
    
    %pat = ':([^:])*:';
    %n = regexpi(str, pat, 'match');
    pat = ':';
    n = regexpi(str, pat, 'split');
    
    for i = 1:numel(n)
        if ~isempty(strfind(n{i}, 'sac'))
            p{end+1} = n{i}; %#ok<*AGROW>
        end
    end
    
end


p(1) = [];



% 
% 
% function [E Acc AUC taskTime cmd OPT status jobOverhead] = searchWithDefaultOptions(classifier, d, pll, timeLimit)
% 
% E = [];
% Acc = [];
% AUC = [];
% status = 0;
% taskTime = 0;
% jobOverhead = 0;
% 
% 
% T = 2;
% [cmd OPT] = sacGetDefaultOptions(classifier);
% [cmds{1:T}] = deal(cmd);
%     
% if pll
%     % if the parallel toolbox is used
%     sched = findResource('scheduler', 'type', 'local');  
%     job = createJob(sched);
%     p = sacInit;
%     set(job, 'FileDependencies', {'sacSuggestFromList.m'});
%     set(job, 'PathDependencies', p);
%     
%     jobTimeLimit = timeLimit/2;
%     
%     createTask(job, @sacSuggestFromList, 4, {cmds, d, classifier});
%     
%     t1 = tic;
%     submit(job);
%     fprintf('   submitting Job %d\n', job.ID);
%     
%     % check for finished jobs and jobs that running longer than TIMEOUT
%     finished = 0;
%     while finished == 0
%         switch job.state
%             case 'finished'
%                 finished = 1;
%                 status = 1;
%                 fprintf('   Job %d finished in %0.5gs\n', job.ID, getRunTime(job));
%                 
%             case 'running'
%                 if ~isempty(job.StartTime)
%                     runningTime = getRunTime(job);
%                     if runningTime > jobTimeLimit
%                         fprintf('Warning: training %s with default options requires more than %0.5gs!\n', classifier, jobTimeLimit);
%                         cancelJob(job, runningTime, jobTimeLimit);
%                         pause(0.1); % added a pause to avoid spamming the job manager
%                         return;
%                     end
%                 end
%         end
%         pause(.1);
%     end
%     tend = toc(t1);
%     
%     % deterimine the time required to execute with default options       
%     out = getAllOutputArguments(job);   
% 
%     E = out{1}(2);
%     Acc = out{2}(2);
%     AUC = out{3}(2);
%     t = out{4};
%     
%     %taskTime = 0.9*t(2); 
%     taskTime = 2*.9*t(2);  % the first job generally takes much longer than
%                            % normal. The 2nd job is still a bit slow, but
%                            % nearly the typical cost. Hyperthreading
%                            % usually makes the job 2x slower.
%     
%     % determine the amount of time lost to overhead of job submission
%     jobOverhead = tend - sum(t);
%     
%     % destroy the job
%     fprintf('   destroying Job %d\n', job.ID);
%     destroy(job);    
% else
%     % serial mode: no parallel toolbox
%     
%     [E Acc AUC t] = sacSuggestFromList(cmds, d, classifier);
%     
%     
%     % the 2nd iteration is still usually a bit slow
%     %taskTime = 0.9 * t(2);
%     taskTime = t(2);
%     
%     jobOverhead = 0;
%     E = E(1);
%     Acc = Acc(1);
%     AUC = AUC(1);
%     status = 1;
% end
% 
% 
% fprintf('   TaskTime = %0.5gs, Job/task overhead = %0.5gs\n', taskTime, jobOverhead);
%     
% 
% 
% 









