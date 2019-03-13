function startPostProcParamSearch(rootDir)

t=tic;
% run parameter searching for 30 minutes by default
timeLim=30;
runParameterSearchForPostprocessing2_parallel_postComp(rootDir,timeLim);