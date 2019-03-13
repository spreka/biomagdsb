function sacDestroyJobs(sched)

global logFID

jlist = sched.Jobs;
if isempty(logFID)
    fprintf('   Destroying %d existing jobs.\n', length(jlist));
else
    sacLog(4,logFID,'   Destroying %d existing jobs.\n', length(jlist));
end

for i = 1:length(jlist)
    destroy(jlist(i));
end