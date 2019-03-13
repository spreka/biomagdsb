function registerNewPipe(newDir)
%Specify a path to a directory that contains the label masks (all the time
%merged masks, i.e. one mask image for one image, and the different nuclei
%are identified by the grayscale id of the image)
%   
%   The newDir new directory must have 2 subfolders a train and a test for
%   all test and train images.    
                
    load('config.mat');
    addpath(genpath(codeBase));
    
    if exist(fullfile(newDir,'train'),'dir')
        trainDir = fullfile(newDir,'train');
        if exist(fullfile(newDir,'test'),'dir')
            testDir = fullfile(newDir,'test');
        else
            error('Error: there must be a test directory below the main folder');
        end
    else
        error('Error: there must be a train directory below the main folder');        
    end
        
    if exist('groundTruthDir','var')
        newScores = evaluation(groundTruthDir,trainDir,newDir,'trainScores.csv');
    end
    
    if exist('testCheckDir','var')
        testCurrentScores = evaluation(testCheckDir,testDir,newDir,'testScores.csv');
    end
    
    if exist('pipeDataBase','var')
        if exist(pipeDataBase,'file')
            load(pipeDataBase); % loads in pipes and scores
            %scores is a map segmentations is a cellarray
        end
        if ~exist('pipes','var')
            pipes = {};
        end
        if ~exist('scores','var')
            scores = containers.Map;
        end
        if ~exist('testScores','var')
            testScores = containers.Map;
        end
        
        pipeID = length(pipes)+1;
        splittedPath = strsplit(newDir,filesep);
        pipes{pipeID}.name = [timeString(5) '_' splittedPath{end}]; % save down as name the folder
        pipes{pipeID}.trainDir = trainDir;
        pipes{pipeID}.testDir = testDir;
        pipes{pipeID}.mean = mean(cell2mat(newScores.values));        
        
        for imageID = newScores.keys
            %train scores
            if isKey(scores,imageID{1})
                scoreForSingle = scores(imageID{1});
            else
                scoreForSingle = [];
            end            
            scoreForSingle(pipeID,1) = newScores(imageID{1});            
            scores(imageID{1}) = scoreForSingle;                        
        end
        for imageID = testCurrentScores.keys
            if isKey(testScores,imageID{1})
                realScoreForSingle = testScores(imageID{1});
            else
                realScoreForSingle = [];
            end            
            realScoreForSingle(pipeID,1) = testCurrentScores(imageID{1});
            testScores(imageID{1}) = realScoreForSingle;
        end
        
        save(pipeDataBase,'scores','pipes','testScores');
                
        [pathToScores,~,~] = fileparts(pipeDataBase);
        headers = getPipeNames(pipes);
        saveScoresToCsv(scores,pathToScores,headers,'mergedTrainScores.csv');
        saveScoresToCsv(testScores,pathToScores,headers,'mergedTestScores.csv');
    end
        
    
end