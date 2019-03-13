function  createMetaSegmentation()
%Creates a merged mask image folder from several pipes' test directory
%based on prediction

load config.mat;
if exist('pipeDataBase','var')
    load(pipeDataBase);
    if ~exist('pipes','var'), return; end
    if ~exist('scores','var'), return; end
    if ~exist('metaTargetDir','var'), return; end
    pipeNames = cell(1,length(pipes));
    for i=1:length(pipes)
        pipeNames{i} = pipes{i}.name;
    end    
    [relevantIndices, ok] = listdlg('ListString',pipeNames,'PromptString','Select pipes to be considered for the meta learner.','ListSize',[400,400]);
    if ~ok, return; end    
    
    for iN = scores.keys
        imageName = iN{1};
        currScores = scores(imageName);
        currScores = currScores(relevantIndices);
        scores(imageName) = currScores; %#ok<AGROW>
    end    
    %delete irrelevant indices
    pipes = pipes(relevantIndices);
    
    targetDir = fullfile(metaTargetDir,[timeString(5) '_MetaLearner']);
    mkdir(targetDir);
    
    metaLearner = BasicWekaClassifier(trainImgDir,scores,2); %WekaRandomForest %NNPredictor(trainImgDir,scores);
    metaLearner.train();
    [predictedScores,out,imageNames] = metaLearner.predict(testImgDir);
    
    save(fullfile(targetDir,'metaLearner.mat'),'metaLearner');
    
    %copy test images into a folder        
    f = fopen(fullfile(targetDir,'origin.csv'),'w');
    for i=1:length(relevantIndices)
        fprintf(f,'%s,',pipeNames{relevantIndices(i)});
    end    
    for i=1:length(out)
        copyfile(fullfile(pipes{out(i)}.testDir,imageNames{i}),fullfile(targetDir,imageNames{i}));    
    end
    
    runLengthEncodeFolder( targetDir );
        
    if exist('testCheckDir','var')
        realScores = evaluation(testCheckDir,targetDir);
    end
    if exist('realScores','var')
        fprintf(f,'\nImageName,selectedPipe,predictedScore,realScore\n');
    else
        fprintf(f,'\nImageName,selectedPipe,predictedScore\n');
    end        
    for i=1:length(out)
        currScores = predictedScores(imageNames{i});        
        if exist('realScores','var')    
            if isKey(realScores,imageNames{i})
                currRealScores = realScores(imageNames{i});            
            else
                currRealScores = 0;
            end
            fprintf(f,'%s,%d,%f,%f\n',imageNames{i},out(i),currScores(out(i)),currRealScores);
        else
            fprintf(f,'%s,%d,%f\n',imageNames{i},out(i),currScores(out(i)));
        end
    end
    fclose(f);  
    
    if exist('realScores','var')
        fprintf('Achieved score: %f compared to estimated theoretical max on train: %f\n',mean(cell2mat(realScores.values)),mean(max(cell2mat(scores.values),[],1)));
        if exist('testScores','var')
            testScoreMatrix = cell2mat(testScores.values);
            testScoreMatrix = testScoreMatrix(relevantIndices,:);
            fprintf('Possible best test score: %f\n',mean(max(testScoreMatrix,[],1)));
        end
    end
    
    runLengthEncodeFolder(targetDir);
    
    movefile(targetDir,[targetDir '_' num2str(mean(cell2mat(realScores.values)),'%.4f')])
end


end

