function [E cmds OPTs status] = sacSuggestRandomSerial(d, classifier, timeLimit, metrics)

global logFID  % the file id of the logfile
global SACROOT



% the first command to generate should be the default
[J.cmds J.OPTs] = sacGetDefaultOptions(classifier);
J.timeLim = timeLimit;
J.logfile = logFID;

t0 = tic;
E = sacSuggestList(J.cmds, d, classifier, metrics, SACROOT, J.logfile);

tElapsed = toc(t0);
tRemain = timeLimit - tElapsed;

taskTime = tElapsed;

i = 2;
while taskTime < tRemain
    
    tElapsed = toc(t0);
    tRemain = timeLimit - tElapsed;
    
    t1 = tic;
    [J(i).cmds J(i).OPTs] = sacGenerateRandomOptions(classifier, 1, 'rand'); %#ok<AGROW>
    J(i).timeLim = tRemain; %#ok<AGROW>
    J(i).logfile = logFID; %#ok<AGROW>
    E(i) = sacSuggestList(J(i).cmds, d, classifier, metrics, SACROOT, J(i).logfile);
    tTime = toc(t1);    
    
    comp = i-1;
    taskTime = (comp/(comp+1))*taskTime + (1/(comp+1)) * tTime;
    sacLog(4,logFID,'    (%d/%d) Job finished in %0.5gs\n', i, i, tTime);
    sacLog(4,logFID,'       --> Updated taskTime = %0.5gs\n', taskTime);
    
    status(i) = 1; %#ok<AGROW>
    
    i = i + 1;
end

tElapsed = toc(t0);
sacLog(3,logFID,' Total elapsed time = %0.5gs\n', tElapsed);

sacLog(4, logFID,'   Retreiving results\n');
cmds = cell(numel(J,1)); OPTs = cell(numel(J),1);
for i = 1:length(J)
    cmds{i} = J(i).cmds{1}; 
    OPTs{i} = J(i).OPTs{1}; 
end


