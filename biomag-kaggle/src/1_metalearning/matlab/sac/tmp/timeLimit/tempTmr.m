

% tmr = timer;
% set(tmr, 'executionMode', 'fixedRate');
% %set(tmr, 'TimerFcn', 'rand(1)');
% set(tmr, 'StartFcn', 'tmpInfLoop');
% set(tmr, 'StartDelay', 10);
% set(tmr, 'TimerFcn', 'stop(tmr)');
% set(tmr, 'StopFcn', 'magic(3)')
% start(tmr)
% 

%%
clear;


sched = findResource('scheduler', 'type', 'local');
j = createJob(sched);
%set(j, 'Timeout', 60);
%createTask(j, @tmpInfLoop, 1, {});
createTask(j, @tempExample1, 1, {});
%createTask(j, @rand, 1, {5});

%alltasks = get(j, 'Tasks');
%set(j, 'CaptureCommandWindowOutput', true)

submit(j);

get(j)
get(j, 'State')

%%
%disp('press space to continue');

wait(j, 'finished', 100);
%pause;


% waitForState(j);
% out = getAllOutputArguments(j);
% disp(out{1});
% destroy(j);

cancel(j);
disp('');
disp('===========================================');
disp('process stopped!');
get(j)

