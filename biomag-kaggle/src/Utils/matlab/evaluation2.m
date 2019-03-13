function scores = evaluation2(groundTruthMap, predictionMap, scoreLocation, scoreFileName)
% A function to calculate the scores in the DataScianceBowl2018
% competition.
% NOTE: All the mask images must be consecutively labeled from 1 to the
% maximum value in the image.
% Author: Abel Szkalisity
%   
%   INPUT:
%       groundTruthMap  containers.Map that stores labelled ground truth
%                       for each input image
%       predictionMap   containers.Map that stores labelled segmentation
%                       for each input image
%       scoreLocation   A path to save the scores.csv file, which contains
%                       the imageIDs and the scores separated by commas. (optional)
%
%   OUTPUT:
%       scores          A containers.map object where key is the string
%                       (the imageID), value is the score

imageIDs = groundTruthMap.keys;

if nargin>2
    saveFile = 1;    
else
    saveFile = 0;
end
if nargin<4
    scoreFileName = 'scores.csv';
end
if saveFile, outFile = fopen(fullfile(scoreLocation,scoreFileName),'w'); end
scores = containers.Map;

nofHashes = 18; currHash = 0; %cmd waitbar
fprintf('Evaluating images:\n');
for i=1:length(imageIDs)
    imageID = imageIDs{i};
    gtMask = groundTruthMap(imageID);
    if isKey(predictionMap, imageID)
          predMask = predictionMap(imageID);
            if (size(predMask,1) == 1)
                score = 0.0;
            else
                score = evalImage(gtMask,predMask);
            end
    else
        score = 0.0;
    end
    scores(imageID) = score;        
    if saveFile, fprintf(outFile,'%s,%f\n',imageID,score); end  
    %waitbar(i/length(imageIDs));
    if (i/length(imageIDs)*nofHashes>currHash), currHash = currHash+1; fprintf('#');  end %cmd waitbar
end
fprintf('\n');

if saveFile,fclose(outFile); end
fprintf('Mean score: %0.4f\n', mean(cell2mat(scores.values)));
if saveFile,disp(['Results are also saved for each image to: ' fullfile(scoreLocation,scoreFileName)]); end

end