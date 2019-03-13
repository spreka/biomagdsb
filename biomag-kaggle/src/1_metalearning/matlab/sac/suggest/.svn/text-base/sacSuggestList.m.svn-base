%function [E t] = sacSuggestList(cmds, d, classifier, criteria, weights, SACROOT, logfile)
function [E t] = sacSuggestList(cmds, d, classifier, M, SACROOT, logfile)
% if logfile contains a filename, it should correspond to the worker
% filename. if it contains an fid, it should be the main logfile



if ischar(logfile)  % this is a worker logfile
    fid = fopen(logfile, 'a');
else  % this is the main logfile
    fid = logfile;
end

N = numel(cmds);
t = zeros(N,1);

status = 0; %#ok<*NASGU>

% check if it is Weka, we need to do a javaclasspathinit
if isempty(strfind('Weka', classifier))
    sacInitJava(SACROOT);
end

% init the evaluation structure E
Eblank = sacEval([],[],d,'Empty');
[E(1:N)] = deal(Eblank);

if ischar(logfile)
    sacLog(5, fid ,'----Starting sacSuggestList Job (%s)\n', logfile);
end

for i = 1:N

    ta=tic;

    str = [classifier ' ' cmds{i}];
    sacLog(5, fid ,'    (%d/%d): %s\n', i, N, str);
    sacLog(5, fid ,'    cross-validating\n');
    
    [y,p,tTr,tPr] = sacCrossValidate(str, d, 5, 1);  %#ok<*ASGLU>
    sacLog(5, fid ,'    evaluating\n');

    %[E(i) s] = sacEvaluate(d.labels, p, '-ACC -waAU -q'); %#ok<NASGU>
    %E(i) = sacEval(d.labels, p, d, 'quiet', criteria{:}, weights); %#ok<AGROW>
	%E(i) = sacEval(d.labels, p, d, 'quiet', 'criteria', criteria, 'weights', weights); 
    E(i) = sacEval(d.labels, p, d, 'quiet', M); 
    
    E(i).trainingTime = tTr; 
    E(i).predictionTime = tPr; 
    
    t(i) = toc(ta);
end



status = 1;

if ischar(logfile)  % this is a worker logfile
    sacLog(5, fid ,'----Job completed\n');
    fclose(fid);
end





