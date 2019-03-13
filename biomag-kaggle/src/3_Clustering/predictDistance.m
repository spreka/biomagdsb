function [ D, imageNames ] = predictDistance( CH, predDir )
%predict pairwise distanse based on the similarity learnt in CH.
%   The function has an inside parameter to see which folder has to be used
%   for the prediction.
%   INPUTS:
%       CH cellarray each entry is a CommonHandles that is trained and can
%       be used in predictClassifier function.
%       Keep in mind to retrain the CH's if the loadBasicFeatures function
%       has been changed to keep consistency between train and prediction.
%   OUTPUT:
%       D: pairwise distance matrix (in a vector form)
%       imageNames: a cellarray of images in the predictionFolder

if nargin<2
    predDir = 'd:\Ábel\SZBK\Projects\Kaggle\Abel\Clustering\test\';
end

nofModels = length(CH);
[features,~,imageNames] = loadBasicFeatures(predDir,0,1);
N = length(features);
nofF = length(features{1});

pairFeatures = zeros(N*(N-1)/2,nofF);

counter = 1;
for i=1:N-1
    for j=i+1:N
        pairFeatures(counter,:) = diffFeature(features{i},features{j});
        counter = counter + 1;
    end
end
probs = cell(1,1,nofModels);

for i=1:nofModels
    try
         [~,probs{1,1,i}] = predictClassifier(pairFeatures,CH{i});
    catch
        load config.mat;
        sacInit(fullfile(codeBase,'1_metalearning','matlab','sac'));         
        sacInit(fullfile(codeBase,'1_metalearning','matlab','sac'));         %do it twice for some stupid reason
        [~,probs{1,1,i}] = predictClassifier(pairFeatures,CH{i});
    end           
end

%create the mean of probabilities
P = mean(cell2mat(probs),3);


%the first class was the similar the 2nd the different ==> The distance is
%the probability of the 2nd class

D = P(:,2);

end

