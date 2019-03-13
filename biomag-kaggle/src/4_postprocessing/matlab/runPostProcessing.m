function runPostProcessing(smallScalePredictionFolder, bigScalePredictionFolder,probMapsFolder,masterResultsFolder,optParametersFile)
% TODO: recheck hard coded parameters
if nargin>4 && exist(optParametersFile,'file')
    paramsData = dlmread(optParametersFile);
    parameters = struct('overlapThresh',paramsData(1),...
                        'probThresh',paramsData(2),...
                        'erosionRadius',paramsData(3),...
                        'dilationRadius',paramsData(4),...
                        'minSize',paramsData(5),...
                        'minOverlap',paramsData(6),...
                        'conn',8);
else
        parameters = struct('minSize',25, 'overlapThresh',0.67, 'probThresh',9547,'erosionRadius',1,'dilationRadius',1,'minOverlap',0.5,'conn',8,'maxVParam',31,'cAreaParam',22);
end

smallScaleImagesFolder = struct('name', smallScalePredictionFolder,'ext','.tiff','scale',1);
bigScaleImagesFolder = struct('name',bigScalePredictionFolder,'ext','.tiff','scale',2);
probMapsFolder = struct('name',probMapsFolder,'ext','.png');
outFolder = struct('name',masterResultsFolder,'ext','.tiff');

postProcessSegmentation_full_parallel( smallScaleImagesFolder, bigScaleImagesFolder, probMapsFolder, outFolder, parameters);
