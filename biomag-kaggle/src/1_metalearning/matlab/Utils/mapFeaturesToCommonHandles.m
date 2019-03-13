function [ CommonHandles ] = mapFeaturesToCommonHandles( features,scores, imageNames)
    CommonHandles.SALT.initialized = 1;
    CommonHandles.TrainingSet.Features = features;
    CommonHandles.ProjectName = 'FakeProject';
    
    class = cell(1,length(features));
    for i=1:length(imageNames)
        currScores = scores(imageNames{i});
        class{i} = (currScores == max(currScores));
    end
    CommonHandles.TrainingSet.Class = class;
end

