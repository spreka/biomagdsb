function resetDatabase()
    load config.mat;
    
    [pathToDataBase,~,~] = fileparts(pipeDataBase);
    
    load(pipeDataBase);
    delete(fullfile(pathToDataBase,'db.mat'));
    delete(fullfile(pathToDataBase,'mergedTestScores.csv'));
    delete(fullfile(pathToDataBase,'mergedTrainScores.csv'));
    
    [basePath,~,~] = fileparts(pipes{1}.trainDir);
    pathBySplit = strsplit(basePath,filesep);
    pathToSegmentations = strjoin(pathBySplit(1:end-1),filesep);
    
    d = dir(pathToSegmentations);
    
    for i=1:length(d)
        if isdir(fullfile(pathToSegmentations,d(i).name)) && ~strcmp(d(i).name,'.') && ~strcmp(d(i).name,'..')
            disp(fullfile(pathToSegmentations,d(i).name));
            registerNewPipe(fullfile(pathToSegmentations,d(i).name));
        end
    end
end