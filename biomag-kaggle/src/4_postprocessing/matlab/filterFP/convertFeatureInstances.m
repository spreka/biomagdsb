function [trainData, labels] = convertFeatureInstances(featuresMap, initialAnnotation)

keys = featuresMap.keys();

trainData = [];
labels = [];

for i=1:length(keys)
    features = featuresMap(keys{i});
    fps = initialAnnotation(keys{i});
    
    % adding all false positives
    trainData = [trainData; features];
    labels = [labels; fps];
end