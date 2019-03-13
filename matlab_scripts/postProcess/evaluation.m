function scores = evaluation(groundTruthDir, predictionDir, scoreLocation, scoreFileName)
% A function to calculate the scores in the DataScianceBowl2018
% competition.
% NOTE: All the mask images must be consecutively labeled from 1 to the
% maximum value in the image.
% Author: Abel Szkalisity
%   
%   INPUT:
%       groundTruthDir  Path to the ground truth mask images. The masks are
%                       stored in one single image, different labels make
%                       difference between masks within the image. Labels
%                       are numbered from 1 to n. 
%       predictionDir   Path to the predicted mask images. The images are
%                       in the same form as above, labeles going from 1->m
%       scoreLocation   A path to save the scores.csv file, which contains
%                       the imageIDs and the scores separated by commas. (optional)
%
%   OUTPUT:
%       scores          A containers.map object where key is the string
%                       (the imageID), value is the score

imageIDs = dir([groundTruthDir filesep '*.tiff']);

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
    imageName = imageIDs(i).name;
    gtMask = imread(fullfile(groundTruthDir,imageName));
    if exist( fullfile(predictionDir,imageName), 'file' )
            predMask = imread(fullfile(predictionDir,imageName));
            if (size(predMask,1) == 1)
                score = 0;
            else
                score = evalImage(gtMask,predMask);
            end
    else
        [~,imageNameExex,~] = fileparts(imageName);        
        ext = '.tiff';
        if exist(fullfile(predictionDir,[imageNameExex ext]), 'file' ),  imageName = [imageNameExex ext]; end
        ext = '.tif';
        if exist(fullfile(predictionDir,[imageNameExex ext]), 'file' ),  imageName = [imageNameExex ext]; end
        ext='.png';
        if exist(fullfile(predictionDir,[imageNameExex ext]), 'file' ),  imageName = [imageNameExex ext]; end
        if exist( fullfile(predictionDir,imageName), 'file' )
            predMask = imread(fullfile(predictionDir,imageName));
            if (size(predMask,1) == 1)
                score = 0;
            else
                score = evalImage(gtMask,predMask);
            end
        else
            score = 0.0;
        end
    end
    scores(imageName) = score;        
    if saveFile, fprintf(outFile,'%s,%f\n',imageName,score); end  
    %waitbar(i/length(imageIDs));
    if (i/length(imageIDs)*nofHashes>currHash), currHash = currHash+1; fprintf('#');  end %cmd waitbar
end
fprintf('\n');

if saveFile,fclose(outFile); end
fprintf('Mean score: %0.4f\n', mean(cell2mat(scores.values)));
if saveFile,disp(['Results are also saved for each image to: ' fullfile(scoreLocation,scoreFileName)]); end

end