function [cmds OPTs E status] = sacGetDistributedJobResults(J,d)


Eblank = sacEval([],[],d,'Empty');
cmds ={};
OPTs = [];
status = [];

ind = 1;
for j = 1:numel(J)
    %results = getAllOutputArguments(J(j).job);
    
    try
        results = getAllOutputArguments(J(j).job);
    catch err
        sacLog(2,logFID,'Failed to get job results! %s\n', err.identifier);
    end
    
    numcmds = length(J(j).cmds);
    for i = 1:numcmds
        if ~isempty(results)
            cmds{ind} = J(j).cmds{i}; %#ok<*AGROW>
            if numcmds > 1
                OPTs{ind} = J(j).OPTs{i};
            else
                OPTs{ind} = J(j).OPTs;
            end
            E(ind) = results{1}(i);
            status(ind) = 1;
            ind = ind + 1;
        else
            cmds{ind} = J(j).cmds{i};
            if numcmds > 1
                OPTs{ind} = J(j).OPTs{i};
            else
                OPTs{ind} = J(j).OPTs;
            end
            E(ind) = Eblank;
            status(ind) = 0;
            ind = ind + 1;
        end
    end
end

